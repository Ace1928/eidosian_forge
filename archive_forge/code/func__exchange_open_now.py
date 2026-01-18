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
def _exchange_open_now(self):
    t = pd.Timestamp.utcnow()
    self._get_exchange_metadata()
    last_day_cutoff = self._get_1y_prices().index[-1] + datetime.timedelta(days=1)
    last_day_cutoff += datetime.timedelta(minutes=20)
    r = t < last_day_cutoff
    return r