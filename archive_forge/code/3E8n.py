# IMPORTING NECESSARY PACKAGES
import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests to a specified URL
import pandas_ta as ta  # For technical analysis indicators
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
from termcolor import colored as cl  # For coloring terminal text
import math  # Provides access to mathematical functions
import logging  # For tracking events that happen when some software runs

# Configuring logging to display information about program execution
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Setting global matplotlib parameters for plot size and style
plt.rcParams["figure.figsize"] = (20, 10)
plt.style.use("fivethirtyeight")


def get_historical_data(symbol: str, start_date: str, interval: str) -> pd.DataFrame:
    """
    Fetches historical stock data for a given symbol from a specified start date at a given interval.

    Parameters:
    - symbol (str): The stock symbol to fetch historical data for.
    - start_date (str): The start date for the historical data in YYYY-MM-DD format.
    - interval (str): The interval for the historical data (e.g., "1d" for daily).

    Returns:
    - pd.DataFrame: A DataFrame containing the historical stock data.
    """
    # Define the URL for the API endpoint
    url = "https://api.benzinga.com/api/v2/bars"
    # Parameters to be sent with the HTTP request
    querystring = {
        "token": "YOUR API KEY",
        "symbols": f"{symbol}",
        "from": f"{start_date}",
        "interval": f"{interval}",
    }

    try:
        # Sending a GET request to the API
        response = requests.get(url, params=querystring)
        # Parsing the JSON response
        hist_json = response.json()
        # Converting the JSON data to a pandas DataFrame
        df = pd.DataFrame(hist_json[0]["candles"])
    except Exception as e:
        logging.error(f"Failed to fetch historical data for {symbol}: {e}")
        raise

    return df


# Example usage of the get_historical_data function
aapl = get_historical_data("AAPL", "1993-01-01", "1W")
logging.info("Fetched historical data for AAPL.")

# CALCULATING DONCHIAN CHANNEL
# Adding Donchian Channel columns to the DataFrame
aapl[["dcl", "dcm", "dcu"]] = aapl.ta.donchian(lower_length=40, upper_length=50)
# Dropping rows with missing values and the 'time' column, renaming 'dateTime' to 'date'
aapl = aapl.dropna().drop("time", axis=1).rename(columns={"dateTime": "date"})
# Setting the 'date' column as the index and converting it to datetime format
aapl = aapl.set_index("date")
aapl.index = pd.to_datetime(aapl.index)

logging.info("Calculated Donchian Channels for AAPL.")

# PLOTTING DONCHIAN CHANNEL
# Plotting the close price and Donchian Channels
plt.plot(aapl[-300:].close, label="CLOSE")
plt.plot(aapl[-300:].dcl, color="black", linestyle="--", alpha=0.3)
plt.plot(aapl[-300:].dcm, color="orange", label="DCM")
plt.plot(aapl[-300:].dcu, color="black", linestyle="--", alpha=0.3, label="DCU,DCL")
plt.legend()
plt.title("AAPL DONCHIAN CHANNELS 50")
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()

logging.info("Plotted Donchian Channels for AAPL.")


def implement_strategy(aapl: pd.DataFrame, investment: float) -> None:
    """
    Implements a trading strategy based on Donchian Channels and prints the results.

    Parameters:
    - aapl (pd.DataFrame): The DataFrame containing AAPL stock data with Donchian Channels.
    - investment (float): The initial investment amount.

    Returns:
    - None
    """
    in_position = False  # Flag to track if currently holding a position
    equity = investment  # Current equity
    no_of_shares = 0  # Number of shares bought

    try:
        for i in range(3, len(aapl)):
            # Buying condition: stock's high equals the upper Donchian Channel and not currently in position
            if aapl["high"][i] == aapl["dcu"][i] and not in_position:
                no_of_shares = math.floor(equity / aapl.close[i])
                equity -= no_of_shares * aapl.close[i]
                in_position = True
                logging.info(
                    f"BUY: {no_of_shares} Shares are bought at ${aapl.close[i]} on {str(aapl.index[i])[:10]}"
                )
            # Selling condition: stock's low equals the lower Donchian Channel and currently in position
            elif aapl["low"][i] == aapl["dcl"][i] and in_position:
                equity += no_of_shares * aapl.close[i]
                in_position = False
                logging.info(
                    f"SELL: {no_of_shares} Shares are sold at ${aapl.close[i]} on {str(aapl.index[i])[:10]}"
                )
        # Closing position at the end if still in position
        if in_position:
            equity += no_of_shares * aapl.close[i]
            logging.info(
                f"\nClosing position at {aapl.close[i]} on {str(aapl.index[i])[:10]}"
            )
            in_position = False

        # Calculating earnings and ROI
        earning = round(equity - investment, 2)
        roi = round(earning / investment * 100, 2)
        logging.info(f"EARNING: ${earning} ; ROI: {roi}%")
    except Exception as e:
        logging.error(f"Error implementing strategy: {e}")
        raise


# Example usage of the implement_strategy function
implement_strategy(aapl, 100000)

# Comparing with SPY ETF buy/hold return
spy = get_historical_data("SPY", "1993-01-01", "1W")
spy_ret = round(((spy.close.iloc[-1] - spy.close.iloc[0]) / spy.close.iloc[0]) * 100, 2)
logging.info(f"SPY ETF buy/hold return: {spy_ret}%")
