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
def _date_to_iloc_linear(dtseries, date, trace=False):
    """Find the location of a date using linear extrapolation.
       Use the endpoints of `dtseries` to calculate the slope
       and yintercept for the line:
           iloc = (slope)*(dtseries) + (yintercept)
       Then use them to calculate the location of `date`
    """
    d1 = _date_to_mdate(dtseries.index[0])
    d2 = _date_to_mdate(dtseries.index[-1])
    if trace:
        print('d1,d2=', d1, d2)
    i1 = 0.0
    i2 = len(dtseries) - 1.0
    if trace:
        print('i1,i2=', i1, i2)
    slope = (i2 - i1) / (d2 - d1)
    yitrcpt1 = i1 - slope * d1
    if trace:
        print('slope,yitrcpt=', slope, yitrcpt1)
    yitrcpt2 = i2 - slope * d2
    if trace:
        print('slope,yitrcpt=', slope, yitrcpt2)
    if yitrcpt1 != yitrcpt2:
        print('WARNING: yintercepts NOT equal!!!(', yitrcpt1, yitrcpt2, ')')
        yitrcpt = (yitrcpt1 + yitrcpt2) / 2.0
    else:
        yitrcpt = yitrcpt1
    return slope * _date_to_mdate(date) + yitrcpt