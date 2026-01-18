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
def _updown_colors(upcolor, downcolor, opens, closes, use_prev_close=False):
    if upcolor == downcolor:
        return [upcolor] * len(opens)
    cmap = {True: upcolor, False: downcolor}
    if not use_prev_close:
        return [cmap[opn < cls] for opn, cls in zip(opens, closes)]
    else:
        first = cmap[opens[0] < closes[0]]
        _list = [cmap[pre < cls] for cls, pre in zip(closes[1:], closes)]
        return [first] + _list