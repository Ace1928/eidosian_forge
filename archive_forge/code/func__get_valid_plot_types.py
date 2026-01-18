import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _get_valid_plot_types(plottype=None):
    _alias_types = {'candlestick': 'candle', 'ohlc_bars': 'ohlc', 'hollow_candle': 'hollow_and_filled', 'hollow': 'hollow_and_filled', 'hnf': 'hollow_and_filled'}
    _valid_types = ['candle', 'ohlc', 'line', 'renko', 'pnf', 'hollow_and_filled']
    _valid_types_all = _valid_types.copy()
    _valid_types_all.extend(_alias_types.keys())
    if plottype is None:
        return _valid_types_all
    elif plottype in _alias_types:
        return _alias_types[plottype]
    return plottype