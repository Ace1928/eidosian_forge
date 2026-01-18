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
def _construct_ohlc_collections(dates, opens, highs, lows, closes, marketcolors=None, config=None):
    """Represent the time, open, high, low, close as a vertical line
    ranging from low to high.  The left tick is the open and the right
    tick is the close.
    *opens*, *highs*, *lows* and *closes* must have the same length.
    NOTE: this code assumes if any value open, high, low, close is
    missing (*-1*) they all are missing

    Parameters
    ----------
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    marketcolors : dict of colors: 'up', 'down'

    Returns
    -------
    ret : list
        a list or tuple of matplotlib collections to be added to the axes
    """
    _check_input(opens, highs, lows, closes)
    if marketcolors is None:
        mktcolors = _get_mpfstyle('classic')['marketcolors']['ohlc']
    else:
        mktcolors = marketcolors['ohlc']
    rangeSegments = [((dt, low), (dt, high)) for dt, low, high in zip(dates, lows, highs)]
    datalen = len(dates)
    avg_dist_between_points = (dates[-1] - dates[0]) / float(datalen)
    ticksize = config['_width_config']['ohlc_ticksize']
    openSegments = [((dt - ticksize, op), (dt, op)) for dt, op in zip(dates, opens)]
    closeSegments = [((dt, close), (dt + ticksize, close)) for dt, close in zip(dates, closes)]
    if mktcolors['up'] == mktcolors['down'] and config['marketcolor_overrides'] is None:
        colors = mktcolors['up']
    else:
        overrides = config['marketcolor_overrides']
        colors = _make_updown_color_list('ohlc', marketcolors, opens, closes, overrides)
    lw = config['_width_config']['ohlc_linewidth']
    rangeCollection = LineCollection(rangeSegments, colors=colors, linewidths=lw)
    openCollection = LineCollection(openSegments, colors=colors, linewidths=lw)
    closeCollection = LineCollection(closeSegments, colors=colors, linewidths=lw)
    return [rangeCollection, openCollection, closeCollection]