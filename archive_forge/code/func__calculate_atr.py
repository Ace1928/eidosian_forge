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
def _calculate_atr(atr_length, highs, lows, closes):
    """Calculate the average true range
    atr_length : time period to calculate over
    all_highs : list of highs
    all_lows : list of lows
    all_closes : list of closes
    """
    if atr_length < 1:
        raise ValueError('Specified atr_length may not be less than 1')
    elif atr_length >= len(closes):
        raise ValueError('Specified atr_length is larger than the length of the dataset: ' + str(len(closes)))
    atr = 0
    for i in range(len(highs) - atr_length, len(highs)):
        high = highs[i]
        low = lows[i]
        close_prev = closes[i - 1]
        tr = max(abs(high - low), abs(high - close_prev), abs(low - close_prev))
        atr += tr
    return atr / atr_length