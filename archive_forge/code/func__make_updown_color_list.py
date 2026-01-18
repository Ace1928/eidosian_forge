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
def _make_updown_color_list(key, marketcolors, opens, closes, overrides=None):
    length = len(opens)
    ups = [marketcolors[key]['up']] * length
    downs = [marketcolors[key]['down']] * length
    if overrides is not None:
        for ix, mco in enumerate(overrides):
            if mco is None:
                continue
            if mcolors.is_color_like(mco):
                ups[ix] = mco
                downs[ix] = mco
            else:
                ups[ix] = mco[key]['up']
                downs[ix] = mco[key]['down']
    return [ups[ix] if opens[ix] < closes[ix] else downs[ix] for ix in range(length)]