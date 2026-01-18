import datetime
import json
import warnings
from collections.abc import MutableMapping
import numpy as _np
import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import quote_summary_valid_modules, _BASE_URL_
from yfinance.exceptions import YFNotImplementedError, YFinanceDataException, YFinanceException
class Quote:

    def __init__(self, data: YfData, symbol: str, proxy=None):
        self._data = data
        self._symbol = symbol
        self.proxy = proxy
        self._info = None
        self._retired_info = None
        self._sustainability = None
        self._recommendations = None
        self._upgrades_downgrades = None
        self._calendar = None
        self._already_scraped = False
        self._already_fetched = False
        self._already_fetched_complementary = False

    @property
    def info(self) -> dict:
        if self._info is None:
            self._fetch_info(self.proxy)
            self._fetch_complementary(self.proxy)
        return self._info

    @property
    def sustainability(self) -> pd.DataFrame:
        if self._sustainability is None:
            raise YFNotImplementedError('sustainability')
        return self._sustainability

    @property
    def recommendations(self) -> pd.DataFrame:
        if self._recommendations is None:
            result = self._fetch(self.proxy, modules=['recommendationTrend'])
            if result is None:
                self._recommendations = pd.DataFrame()
            else:
                try:
                    data = result['quoteSummary']['result'][0]['recommendationTrend']['trend']
                except (KeyError, IndexError):
                    raise YFinanceDataException(f'Failed to parse json response from Yahoo Finance: {result}')
                self._recommendations = pd.DataFrame(data)
        return self._recommendations

    @property
    def upgrades_downgrades(self) -> pd.DataFrame:
        if self._upgrades_downgrades is None:
            result = self._fetch(self.proxy, modules=['upgradeDowngradeHistory'])
            if result is None:
                self._upgrades_downgrades = pd.DataFrame()
            else:
                try:
                    data = result['quoteSummary']['result'][0]['upgradeDowngradeHistory']['history']
                    if len(data) == 0:
                        raise YFinanceDataException(f'No upgrade/downgrade history found for {self._symbol}')
                    df = pd.DataFrame(data)
                    df.rename(columns={'epochGradeDate': 'GradeDate', 'firm': 'Firm', 'toGrade': 'ToGrade', 'fromGrade': 'FromGrade', 'action': 'Action'}, inplace=True)
                    df.set_index('GradeDate', inplace=True)
                    df.index = pd.to_datetime(df.index, unit='s')
                    self._upgrades_downgrades = df
                except (KeyError, IndexError):
                    raise YFinanceDataException(f'Failed to parse json response from Yahoo Finance: {result}')
        return self._upgrades_downgrades

    @property
    def calendar(self) -> dict:
        if self._calendar is None:
            self._fetch_calendar()
        return self._calendar

    @staticmethod
    def valid_modules():
        return quote_summary_valid_modules

    def _fetch(self, proxy, modules: list):
        if not isinstance(modules, list):
            raise YFinanceException('Should provide a list of modules, see available modules using `valid_modules`')
        modules = ','.join([m for m in modules if m in quote_summary_valid_modules])
        if len(modules) == 0:
            raise YFinanceException('No valid modules provided, see available modules using `valid_modules`')
        params_dict = {'modules': modules, 'corsDomain': 'finance.yahoo.com', 'formatted': 'false', 'symbol': self._symbol}
        try:
            result = self._data.get_raw_json(_QUOTE_SUMMARY_URL_ + f'/{self._symbol}', user_agent_headers=self._data.user_agent_headers, params=params_dict, proxy=proxy)
        except requests.exceptions.HTTPError as e:
            utils.get_yf_logger().error(str(e))
            return None
        return result

    def _fetch_info(self, proxy):
        if self._already_fetched:
            return
        self._already_fetched = True
        modules = ['financialData', 'quoteType', 'defaultKeyStatistics', 'assetProfile', 'summaryDetail']
        result = self._fetch(proxy, modules=modules)
        if result is None:
            self._info = {}
            return
        result['quoteSummary']['result'][0]['symbol'] = self._symbol
        query1_info = next((info for info in result.get('quoteSummary', {}).get('result', []) if info['symbol'] == self._symbol), None)
        for k in query1_info:
            if 'maxAge' in query1_info[k] and query1_info[k]['maxAge'] == 1:
                query1_info[k]['maxAge'] = 86400
        query1_info = {k1: v1 for k, v in query1_info.items() if isinstance(v, dict) for k1, v1 in v.items() if v1}

        def _format(k, v):
            if isinstance(v, dict) and 'raw' in v and ('fmt' in v):
                v2 = v['fmt'] if k in {'regularMarketTime', 'postMarketTime'} else v['raw']
            elif isinstance(v, list):
                v2 = [_format(None, x) for x in v]
            elif isinstance(v, dict):
                v2 = {k: _format(k, x) for k, x in v.items()}
            elif isinstance(v, str):
                v2 = v.replace('\xa0', ' ')
            else:
                v2 = v
            return v2
        for k, v in query1_info.items():
            query1_info[k] = _format(k, v)
        self._info = query1_info

    def _fetch_complementary(self, proxy):
        if self._already_fetched_complementary:
            return
        self._already_fetched_complementary = True
        self._fetch_info(proxy)
        if self._info is None:
            return
        keys = {'trailingPegRatio'}
        if keys:
            url = f'https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{self._symbol}?symbol={self._symbol}'
            for k in keys:
                url += '&type=' + k
            start = pd.Timestamp.utcnow().floor('D') - datetime.timedelta(days=365 // 2)
            start = int(start.timestamp())
            end = pd.Timestamp.utcnow().ceil('D')
            end = int(end.timestamp())
            url += f'&period1={start}&period2={end}'
            json_str = self._data.cache_get(url=url, proxy=proxy).text
            json_data = json.loads(json_str)
            json_result = json_data.get('timeseries') or json_data.get('finance')
            if json_result['error'] is not None:
                raise YFinanceException('Failed to parse json response from Yahoo Finance: ' + str(json_result['error']))
            for k in keys:
                keydict = json_result['result'][0]
                if k in keydict:
                    self._info[k] = keydict[k][-1]['reportedValue']['raw']
                else:
                    self.info[k] = None

    def _fetch_calendar(self):
        result = self._fetch(self.proxy, modules=['calendarEvents'])
        if result is None:
            self._calendar = {}
            return
        try:
            self._calendar = dict()
            _events = result['quoteSummary']['result'][0]['calendarEvents']
            if 'dividendDate' in _events:
                self._calendar['Dividend Date'] = datetime.datetime.fromtimestamp(_events['dividendDate']).date()
            if 'exDividendDate' in _events:
                self._calendar['Ex-Dividend Date'] = datetime.datetime.fromtimestamp(_events['exDividendDate']).date()
            earnings = _events.get('earnings')
            if earnings is not None:
                self._calendar['Earnings Date'] = [datetime.datetime.fromtimestamp(d).date() for d in earnings.get('earningsDate', [])]
                self._calendar['Earnings High'] = earnings.get('earningsHigh', None)
                self._calendar['Earnings Low'] = earnings.get('earningsLow', None)
                self._calendar['Earnings Average'] = earnings.get('earningsAverage', None)
                self._calendar['Revenue High'] = earnings.get('revenueHigh', None)
                self._calendar['Revenue Low'] = earnings.get('revenueLow', None)
                self._calendar['Revenue Average'] = earnings.get('revenueAverage', None)
        except (KeyError, IndexError):
            raise YFinanceDataException(f'Failed to parse json response from Yahoo Finance: {result}')