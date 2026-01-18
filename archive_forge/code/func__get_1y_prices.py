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
def _get_1y_prices(self, fullDaysOnly=False):
    if self._prices_1y is None:
        self._prices_1y = self._tkr.history(period='380d', auto_adjust=False, keepna=True, proxy=self.proxy)
        self._md = self._tkr.get_history_metadata(proxy=self.proxy)
        try:
            ctp = self._md['currentTradingPeriod']
            self._today_open = pd.to_datetime(ctp['regular']['start'], unit='s', utc=True).tz_convert(self.timezone)
            self._today_close = pd.to_datetime(ctp['regular']['end'], unit='s', utc=True).tz_convert(self.timezone)
            self._today_midnight = self._today_close.ceil('D')
        except Exception:
            self._today_open = None
            self._today_close = None
            self._today_midnight = None
            raise
    if self._prices_1y.empty:
        return self._prices_1y
    dnow = pd.Timestamp.utcnow().tz_convert(self.timezone).date()
    d1 = dnow
    d0 = d1 + datetime.timedelta(days=1) - utils._interval_to_timedelta('1y')
    if fullDaysOnly and self._exchange_open_now():
        d1 -= utils._interval_to_timedelta('1d')
    return self._prices_1y.loc[str(d0):str(d1)]