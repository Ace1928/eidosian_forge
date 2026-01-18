import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
def _fetch_and_parse(self):
    try:
        result = self._fetch(self.proxy)
    except requests.exceptions.HTTPError as e:
        utils.get_yf_logger().error(str(e))
        self._major = pd.DataFrame()
        self._major_direct_holders = pd.DataFrame()
        self._institutional = pd.DataFrame()
        self._mutualfund = pd.DataFrame()
        self._insider_transactions = pd.DataFrame()
        self._insider_purchases = pd.DataFrame()
        self._insider_roster = pd.DataFrame()
        return
    try:
        data = result['quoteSummary']['result'][0]
        self._parse_institution_ownership(data['institutionOwnership'])
        self._parse_fund_ownership(data['fundOwnership'])
        self._parse_major_holders_breakdown(data['majorHoldersBreakdown'])
        self._parse_insider_transactions(data['insiderTransactions'])
        self._parse_insider_holders(data['insiderHolders'])
        self._parse_net_share_purchase_activity(data['netSharePurchaseActivity'])
    except (KeyError, IndexError):
        raise YFinanceDataException('Failed to parse holders json data.')