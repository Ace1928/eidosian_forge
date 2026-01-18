import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
def _parse_insider_transactions(self, data):
    holders = data['transactions']
    for owner in holders:
        for k, v in owner.items():
            owner[k] = self._parse_raw_values(v)
        del owner['maxAge']
    df = pd.DataFrame(holders)
    if not df.empty:
        df['startDate'] = pd.to_datetime(df['startDate'], unit='s')
        df.rename(columns={'startDate': 'Start Date', 'filerName': 'Insider', 'filerRelation': 'Position', 'filerUrl': 'URL', 'moneyText': 'Transaction', 'transactionText': 'Text', 'shares': 'Shares', 'value': 'Value', 'ownership': 'Ownership'}, inplace=True)
    self._insider_transactions = df