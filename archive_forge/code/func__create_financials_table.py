import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
def _create_financials_table(self, name, timescale, proxy):
    if name == 'income':
        name = 'financials'
    keys = const.fundamentals_keys[name]
    try:
        return self.get_financials_time_series(timescale, keys, proxy)
    except Exception:
        pass