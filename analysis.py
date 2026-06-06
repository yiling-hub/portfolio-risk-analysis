import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Download price data
tickers = ['AAPL', 'MSFT', 'TSLA', 'SPY']
start_date = '2020-01-01'
end_date = '2025-12-31'

prices = yf.download(tickers, start=start_date, end=end_date)['Close']

# Step 2: Calculate daily returns
returns = prices.pct_change().dropna()

print(returns)

# Step 3: Define portfolio weights
weights = np.array([0.30, 0.25, 0.20, 0.25])

# Step 4: Calculate portfolio daily returns
portfolio_returns = returns.dot(weights)

# Step 5: Calculate key risk metrics
avg_daily_return = portfolio_returns.mean()
daily_volatility = portfolio_returns.std()
annual_return = avg_daily_return * 252
annual_volatility = daily_volatility * np.sqrt(252)

print(f"Average daily return : {avg_daily_return:.4f}")
print(f"Daily volatility     : {daily_volatility:.4f}")
print(f"Annualized return    : {annual_return:.4f}")
print(f"Annualized volatility: {annual_volatility:.4f}")

# Step 6: Value at Risk (VaR)
confidence_level = 0.95

# Parametric VaR (assumes normal distribution)
parametric_var = stats.norm.ppf(1 - confidence_level,
                                avg_daily_return,
                                daily_volatility)

# Historical VaR (uses actual data)
historical_var = np.percentile(portfolio_returns,
                               (1 - confidence_level) * 100)

print(f"\n--- Value at Risk (1-day, 95% confidence) ---")
print(f"Parametric VaR : {parametric_var:.4f} ({parametric_var*100:.2f}%)")
print(f"Historical VaR : {historical_var:.4f} ({historical_var*100:.2f}%)")

# Step 7: Stress Testing
print(f"\n--- Stress Test: COVID Crash ---")

# Filter returns for the COVID crash period
covid_crash = portfolio_returns['2020-02-20':'2020-03-23']
covid_loss = covid_crash.sum()

print(f"Portfolio loss during COVID crash: {covid_loss*100:.2f}%")
print(f"Number of trading days           : {len(covid_crash)}")
print(f"Worst single day during crash    : {covid_crash.min()*100:.2f}%")

# Step 8: Visualizations
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Chart 1: Cumulative portfolio growth
cumulative = (1 + portfolio_returns).cumprod()
axes[0].plot(cumulative.index, cumulative.values,
             color='steelblue', linewidth=1.5)
axes[0].set_title('Cumulative Portfolio Growth (2020-2025)')
axes[0].set_ylabel('Growth of $1 invested')
axes[0].axhline(y=1, color='gray', linestyle='--', linewidth=0.8)

# Chart 2: Daily returns distribution
axes[1].hist(portfolio_returns, bins=80, color='steelblue',
             edgecolor='white', linewidth=0.3)
axes[1].axvline(x=parametric_var, color='red', linestyle='--',
                linewidth=1.5, label=f'Parametric VaR ({parametric_var*100:.2f}%)')
axes[1].axvline(x=historical_var, color='orange', linestyle='--',
                linewidth=1.5, label=f'Historical VaR ({historical_var*100:.2f}%)')
axes[1].set_title('Daily Returns Distribution with VaR Thresholds')
axes[1].set_xlabel('Daily Return')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Chart 3: Rolling volatility
rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
axes[2].plot(rolling_vol.index, rolling_vol.values,
             color='coral', linewidth=1.2)
axes[2].set_title('Rolling 30-Day Annualized Volatility')
axes[2].set_ylabel('Volatility')
axes[2].set_xlabel('Date')

plt.tight_layout()
plt.savefig('portfolio_risk_analysis.png', dpi=150, bbox_inches='tight')
print("\nChart saved as portfolio_risk_analysis.png")
