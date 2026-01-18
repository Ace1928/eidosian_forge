import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
def _parse_insider_holders(self, data):
    holders = data['holders']
    for owner in holders:
        for k, v in owner.items():
            owner[k] = self._parse_raw_values(v)
        del owner['maxAge']
    df = pd.DataFrame(holders)
    if not df.empty:
        df['positionDirectDate'] = pd.to_datetime(df['positionDirectDate'], unit='s')
        df['latestTransDate'] = pd.to_datetime(df['latestTransDate'], unit='s')
        df.rename(columns={'name': 'Name', 'relation': 'Position', 'url': 'URL', 'transactionDescription': 'Most Recent Transaction', 'latestTransDate': 'Latest Transaction Date', 'positionDirectDate': 'Position Direct Date', 'positionDirect': 'Shares Owned Directly', 'positionIndirectDate': 'Position Indirect Date', 'positionIndirect': 'Shares Owned Indirectly'}, inplace=True)
        df['Name'] = df['Name'].astype(str)
        df['Position'] = df['Position'].astype(str)
        df['URL'] = df['URL'].astype(str)
        df['Most Recent Transaction'] = df['Most Recent Transaction'].astype(str)
    self._insider_roster = df