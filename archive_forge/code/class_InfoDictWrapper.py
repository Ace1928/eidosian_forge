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
class InfoDictWrapper(MutableMapping):
    """ Simple wrapper around info dict, intercepting 'gets' to
    print how-to-migrate messages for specific keys. Requires
    override dict API"""

    def __init__(self, info):
        self.info = info

    def keys(self):
        return self.info.keys()

    def __str__(self):
        return self.info.__str__()

    def __repr__(self):
        return self.info.__repr__()

    def __contains__(self, k):
        return k in self.info.keys()

    def __getitem__(self, k):
        if k in info_retired_keys_price:
            warnings.warn(f"Price data removed from info (key='{k}'). Use Ticker.fast_info or history() instead", DeprecationWarning)
            return None
        elif k in info_retired_keys_exchange:
            warnings.warn(f"Exchange data removed from info (key='{k}'). Use Ticker.fast_info or Ticker.get_history_metadata() instead", DeprecationWarning)
            return None
        elif k in info_retired_keys_marketCap:
            warnings.warn(f"Market cap removed from info (key='{k}'). Use Ticker.fast_info instead", DeprecationWarning)
            return None
        elif k in info_retired_keys_symbol:
            warnings.warn(f"Symbol removed from info (key='{k}'). You know this already", DeprecationWarning)
            return None
        return self.info[self._keytransform(k)]

    def __setitem__(self, k, value):
        self.info[self._keytransform(k)] = value

    def __delitem__(self, k):
        del self.info[self._keytransform(k)]

    def __iter__(self):
        return iter(self.info)

    def __len__(self):
        return len(self.info)

    def _keytransform(self, k):
        return k