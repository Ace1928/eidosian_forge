from __future__ import print_function
from io import StringIO
import json as _json
import warnings
from typing import Optional, Union
from urllib.parse import quote as urlencode
import pandas as pd
import requests
from . import utils, cache
from .data import YfData
from .scrapers.analysis import Analysis
from .scrapers.fundamentals import Fundamentals
from .scrapers.holders import Holders
from .scrapers.quote import Quote, FastInfo
from .scrapers.history import PriceHistory
from .const import _BASE_URL_, _ROOT_URL_
def _lazy_load_price_history(self):
    if self._price_history is None:
        self._price_history = PriceHistory(self._data, self.ticker, self._get_ticker_tz(self.proxy, timeout=10))
    return self._price_history