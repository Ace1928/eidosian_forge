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
def _date_to_iloc(dtseries, date):
    """Convert a `date` to a location, given a date series w/a datetime index.
       If `date` does not exactly match a date in the series then interpolate between two dates.
       If `date` is outside the range of dates in the series, then raise an exception
      .
    """
    d1s = dtseries.loc[date:]
    if len(d1s) < 1:
        sdtrange = str(dtseries[0]) + ' to ' + str(dtseries[-1])
        raise ValueError('User specified line date "' + str(date) + '" is beyond (greater than) range of plotted data (' + sdtrange + ').')
    d1 = d1s.index[0]
    d2s = dtseries.loc[:date]
    if len(d2s) < 1:
        sdtrange = str(dtseries[0]) + ' to ' + str(dtseries[-1])
        raise ValueError('User specified line date "' + str(date) + '" is before (less than) range of plotted data (' + sdtrange + ').')
    d2 = dtseries.loc[:date].index[-1]
    loc1 = dtseries.index.get_loc(d1)
    if isinstance(loc1, slice):
        loc1 = loc1.start
    loc2 = dtseries.index.get_loc(d2)
    if isinstance(loc2, slice):
        loc2 = loc2.stop - 1
    return (loc1 + loc2) / 2.0