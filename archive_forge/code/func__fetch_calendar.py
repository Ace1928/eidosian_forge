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