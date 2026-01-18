import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
def _parse_net_share_purchase_activity(self, data):
    df = pd.DataFrame({'Insider Purchases Last ' + data.get('period', ''): ['Purchases', 'Sales', 'Net Shares Purchased (Sold)', 'Total Insider Shares Held', '% Net Shares Purchased (Sold)', '% Buy Shares', '% Sell Shares'], 'Shares': [data.get('buyInfoShares'), data.get('sellInfoShares'), data.get('netInfoShares'), data.get('totalInsiderShares'), data.get('netPercentInsiderShares'), data.get('buyPercentInsiderShares'), data.get('sellPercentInsiderShares')], 'Trans': [data.get('buyInfoCount'), data.get('sellInfoCount'), data.get('netInfoCount'), pd.NA, pd.NA, pd.NA, pd.NA]}).convert_dtypes()
    self._insider_purchases = df