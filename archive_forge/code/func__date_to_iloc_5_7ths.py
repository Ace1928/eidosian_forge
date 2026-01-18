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
def _date_to_iloc_5_7ths(dtseries, date, direction, trace=False):
    first = _date_to_mdate(dtseries.index[0])
    last = _date_to_mdate(dtseries.index[-1])
    avg_days_between_points = (last - first) / float(len(dtseries))
    if avg_days_between_points < 0.33:
        return None
    if direction == 'forward':
        delta = _date_to_mdate(date) - _date_to_mdate(dtseries.index[-1])
        loc_5_7ths = len(dtseries) - 1 + 5 / 7.0 * delta
    elif direction == 'backward':
        delta = _date_to_mdate(dtseries.index[0]) - _date_to_mdate(date)
        loc_5_7ths = -(5.0 / 7.0) * delta
    else:
        raise ValueError('_date_to_iloc_5_7ths got BAD direction value=' + str(direction))
    return loc_5_7ths