import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
def _parse_major_holders_breakdown(self, data):
    if 'maxAge' in data:
        del data['maxAge']
    df = pd.DataFrame.from_dict(data, orient='index')
    if not df.empty:
        df.columns.name = 'Breakdown'
        df.rename(columns={df.columns[0]: 'Value'}, inplace=True)
    self._major = df