# IMPORTING PACKAGES

import pandas as pd
import requests
import pandas_ta as ta
import matplotlib.pyplot as plt
from termcolor import colored as cl
import math 

plt.rcParams['figure.figsize'] = (20,10)
plt.style.use('fivethirtyeight')
If you haven’t installed any of the imported packages, make sure to do so using the pip command in your terminal.

Extracting Historical Data
We are going to backtest our breakout strategy on Apple’s stock. So in order to obtain the historical stock data of Apple, we are going to use Benzinga’s Historical Bar Data API endpoint. The following Python code uses the endpoint to extract Apple’s stock data from 1993:

# EXTRACTING HISTORICAL DATA

def get_historical_data(symbol, start_date, interval):
    url = "https://api.benzinga.com/api/v2/bars"
    querystring = {"token":"YOUR API KEY","symbols":f"{symbol}","from":f"{start_date}","interval":f"{interval}"}

    hist_json = requests.get(url, params = querystring).json()
    df = pd.DataFrame(hist_json[0]['candles'])
    
    return df

aapl = get_historical_data('AAPL', '1993-01-01', '1W')
aapl.tail()