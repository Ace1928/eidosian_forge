import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
@staticmethod
def _parse_raw_values(data):
    if isinstance(data, dict) and 'raw' in data:
        return data['raw']
    return data