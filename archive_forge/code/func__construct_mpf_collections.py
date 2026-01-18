import numpy  as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from itertools import cycle
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.patches     import Ellipse
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _alines_validator, _bypass_kwarg_validation
from mplfinance._arg_validators import _xlim_validator, _is_datelike
from mplfinance._styles         import _get_mpfstyle
from mplfinance._helpers        import _mpf_to_rgba
from six.moves import zip
from matplotlib.ticker import Formatter
def _construct_mpf_collections(ptype, dates, xdates, opens, highs, lows, closes, volumes, config, style):
    collections = None
    if ptype == 'candle' or ptype == 'candlestick':
        collections = _construct_candlestick_collections(xdates, opens, highs, lows, closes, marketcolors=style['marketcolors'], config=config)
    elif ptype == 'hollow_and_filled':
        collections = _construct_hollow_candlestick_collections(xdates, opens, highs, lows, closes, marketcolors=style['marketcolors'], config=config)
    elif ptype == 'ohlc' or ptype == 'bars' or ptype == 'ohlc_bars':
        collections = _construct_ohlc_collections(xdates, opens, highs, lows, closes, marketcolors=style['marketcolors'], config=config)
    elif ptype == 'renko':
        collections = _construct_renko_collections(dates, highs, lows, volumes, config['renko_params'], closes, marketcolors=style['marketcolors'])
    elif ptype == 'pnf':
        raise ValueError('Plot type="pnf" should no longer come this way!')
    else:
        raise TypeError('Unknown ptype="', str(ptype), '"')
    return collections