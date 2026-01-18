import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import ccxt  # For connecting to various trading exchanges
import backtrader as bt  # For backtesting
import asyncio
import aiohttp
import websocket
import logging
import yfinance as yf  # For downloading market data from Yahoo Finance

# Configuration for CCXT to connect to an exchange
api_key: str = input("Enter your API key: ")
secret_key: str = input("Enter your secret key: ")
exchange = ccxt.binance(
    {
        "apiKey": api_key,
        "secret": secret_key,
        "enableRateLimit": True,
    }
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define the trading bot class
class TradingBot:
    def __init__(self, simulation_mode: bool = False):
        self.data: pd.DataFrame = pd.DataFrame()
        self.strategy: str = ""
        self.simulation_mode = simulation_mode
        logging.info("TradingBot initialized with no data and strategy.")

    def fetch_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, source: str = "ccxt"
    ) -> pd.DataFrame:
        """Fetch historical price data from exchange or Yahoo Finance."""
        logging.info(
            f"Fetching data for {symbol} with timeframe {timeframe}, limit {limit}, from {source}."
        )
        if source == "ccxt":
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        elif source == "yfinance":
            df = yf.download(symbol, period=f"{limit}d", interval=timeframe)
            df.reset_index(inplace=True)
            df.rename(
                columns={
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                inplace=True,
            )
        self.data = df
        logging.info("Data fetching complete.")
        return df

    def apply_strategy(self, strategy_type: str, **kwargs) -> None:
        """Apply trading strategy based on type."""
        logging.info(f"Applying {strategy_type} strategy.")
        if strategy_type == "mean_reversion":
            window = kwargs.get("window", 30)
            self.data["moving_average"] = (
                self.data["close"].rolling(window=window).mean()
            )
            self.data["distance_from_mean"] = (
                self.data["close"] / self.data["moving_average"] - 1
            )
            self.data["entry"] = self.data["distance_from_mean"] < -0.05
            self.data["exit"] = self.data["distance_from_mean"] > 0.05
        elif strategy_type == "momentum":
            self.data["momentum"] = talib.MOM(self.data["close"], timeperiod=10)
            self.data["buy_signal"] = self.data["momentum"] > 100
            self.data["sell_signal"] = self.data["momentum"] < -100
        elif strategy_type == "scalping":
            self.data["price_diff"] = self.data["close"].diff()
            self.data["scalp_entry"] = self.data["price_diff"] > 0
            self.data["scalp_exit"] = self.data["price_diff"] < 0
        elif strategy_type == "machine_learning":
            features = self.data[["open", "high", "low", "close", "volume"]]
            target = (self.data["close"].shift(-1) > self.data["close"]).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            logging.info(f"Machine Learning strategy accuracy: {accuracy:.2f}")
        logging.info(f"{strategy_type.capitalize()} strategy applied.")

    async def async_fetch_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, source: str = "ccxt"
    ) -> pd.DataFrame:
        """Fetch historical price data from exchange asynchronously."""
        logging.info(
            f"Asynchronously fetching data for {symbol} with timeframe {timeframe}, limit {limit}, from {source}."
        )
        if source == "ccxt":
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    df = pd.DataFrame(
                        data,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        elif source == "yfinance":
            df = await asyncio.to_thread(
                yf.download, symbol, period=f"{limit}d", interval=timeframe
            )
            df.reset_index(inplace=True)
            df.rename(
                columns={
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                inplace=True,
            )
        self.data = df
        logging.info("Asynchronous data fetching complete.")
        return df

    def optimize_model(self, features: pd.DataFrame, target: pd.Series) -> dict:
        """Optimize machine learning model parameters."""
        logging.info("Optimizing machine learning model parameters.")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_features": ["auto", "sqrt", "log2"],
            "max_depth": [4, 5, 6, 7, 8],
            "criterion": ["gini", "entropy"],
        }
        rfc = RandomForestClassifier(random_state=42)
        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        CV_rfc.fit(features, target)
        best_params = CV_rfc.best_params_
        logging.info(f"Optimal parameters found: {best_params}")
        return best_params

    def start_websocket(self) -> None:
        """Start websocket for real-time data processing."""
        logging.info("Starting websocket for real-time data processing.")

        def on_message(ws, message):
            logging.info(f"Websocket message received: {message}")
            # Real-time trading logic here

        def on_error(ws, error):
            logging.error(f"Websocket error: {error}")

        def on_close(ws):
            logging.info("Websocket closed.")

        websocket.enableTrace(True)
        ws = websocket.WebSocketApp(
            "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever()

    def adjust_risk_parameters(self) -> None:
        """Adjust trading risk parameters based on recent volatility."""
        logging.info("Adjusting risk parameters based on recent volatility.")
        recent_volatility = np.std(self.data["close"][-10:])
        threshold = 0.05  # Example threshold
        if recent_volatility > threshold:
            risk_level = "High"
            logging.info(f"Risk level set to {risk_level}. Reducing position size.")
            # Reduce position size
        else:
            risk_level = "Low"
            logging.info(f"Risk level set to {risk_level}. Increasing position size.")
            # Increase position size

    def test_strategies(self, symbols: list, strategy: str):
        """Test a strategy across multiple symbols."""
        results = {}
        for symbol in symbols:
            self.fetch_data(symbol)
            self.apply_strategy(strategy)
            # Assuming there's a method to calculate performance
            results[symbol] = self.evaluate_performance()
        return results

    def evaluate_performance(self):
        """Evaluate the performance of the applied strategy."""
        # Dummy implementation
        return np.random.random()

    def backtest_with_costs(self, strategy) -> None:
        """Backtest a given strategy including transaction costs."""
        logging.info("Backtesting strategy with transaction costs.")
        transaction_costs = 0.1  # 0.1% of the trade amount
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy)
        data = bt.feeds.PandasData(dataname=self.data)
        cerebro.adddata(data)
        cerebro.run()
        self.data["net_profit"] = self.data["profit"] - (
            self.data["trade_amount"] * transaction_costs / 100
        )
        cerebro.plot()
        logging.info("Backtesting complete.")


# Main function to run the bot
if __name__ == "__main__":
    bot = TradingBot(simulation_mode=True)
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Example symbols
    asyncio.run(bot.async_fetch_data("BTC/USDT"))
    bot.fetch_data("BTC/USDT")
    bot.apply_strategy("mean_reversion")
    bot.apply_strategy("momentum")
    bot.apply_strategy("scalping")
    bot.apply_strategy("machine_learning")
    bot.optimize_model()
    bot.start_websocket()
    bot.adjust_risk_parameters()
    bot.backtest_with_costs(bt.strategies.MA_CrossOver)


# Main function to run the bot in simulation mode
if __name__ == "__main__":
    bot = TradingBot(simulation_mode=True)
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Example symbols
    strategy = "mean_reversion"
    results = bot.test_strategies(symbols, strategy)
    print(results)
