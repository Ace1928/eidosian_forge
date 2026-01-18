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
def _tline_point_to_point(dfslice, tline_use):
    p1 = dfslice.iloc[0]
    p2 = dfslice.iloc[-1]
    x1 = p1.name
    y1 = p1[tline_use].mean()
    x2 = p2.name
    y2 = p2[tline_use].mean()
    return ((x1, y1), (x2, y2))