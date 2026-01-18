import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _is_marketcolor_object(obj):
    if not isinstance(obj, dict):
        return False
    market_colors_keys = ('candle', 'edge', 'wick', 'ohlc')
    return all([k in obj for k in market_colors_keys])