from __future__ import print_function
import datetime as _datetime
from collections import namedtuple as _namedtuple
import pandas as _pd
from .base import TickerBase
from .const import _BASE_URL_
def _options2df(self, opt, tz=None):
    data = _pd.DataFrame(opt).reindex(columns=['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency'])
    data['lastTradeDate'] = _pd.to_datetime(data['lastTradeDate'], unit='s', utc=True)
    if tz is not None:
        data['lastTradeDate'] = data['lastTradeDate'].dt.tz_convert(tz)
    return data