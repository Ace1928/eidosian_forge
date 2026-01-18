import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
@utils.log_indent_decorator
def _fix_unit_switch(self, df, interval, tz_exchange):
    return self._fix_prices_sudden_change(df, interval, tz_exchange, 100.0)