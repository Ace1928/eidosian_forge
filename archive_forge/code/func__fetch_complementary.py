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