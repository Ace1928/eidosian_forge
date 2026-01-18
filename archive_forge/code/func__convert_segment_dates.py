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
def _convert_segment_dates(segments, dtindex):
    """
    Convert line segment dates to matplotlib dates
    Inputted segment dates may be: pandas-parseable date-time string, pandas timestamp,
                                   or a python datetime or date, or (if dtindex is not None) integer index
    A "segment" is a "sequence of lines",
        see: https://matplotlib.org/api/collections_api.html#matplotlib.collections.LineCollection
    """
    if dtindex is not None:
        dtseries = dtindex.to_series()
    converted = []
    for line in segments:
        new_line = []
        for dt, value in line:
            if dtindex is not None:
                date = _date_to_iloc(dtseries, dt)
            else:
                date = _date_to_mdate(dt)
            if date is None:
                raise TypeError('NON-DATE in segment line=' + str(line))
            new_line.append((date, value))
        converted.append(new_line)
    return converted