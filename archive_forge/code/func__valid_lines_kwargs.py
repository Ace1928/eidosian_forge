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
def _valid_lines_kwargs():
    """
    Construct and return the "valid lines (hlines,vlines,alines,tlines) kwargs table"
    for the mplfinance.plot() `[h|v|a|t]lines=` kwarg functions.
    A valid kwargs table is a `dict` of `dict`s. The keys of the outer dict are
    the valid key-words for the function.  The value for each key is a dict containing 3
    specific keys: "Default", "Description" and "Validator" with the following values:
        "Default"      - The default value for the kwarg if none is specified.
        "Description"  - The description for the kwarg.
        "Validator"    - A function that takes the caller specified value for the kwarg,
                         and validates that it is the correct type, and (for kwargs with
                         a limited set of allowed values) may also validate that the
                         kwarg value is one of the allowed values.
    """
    valid_linestyles = ['-', 'solid', '--', 'dashed', '-.', 'dashdot', ':', 'dotted', None, ' ', '']
    vkwargs = {'hlines': {'Default': None, 'Description': 'Draw one or more HORIZONTAL LINES across entire plot, by' + ' specifying a price, or sequence of prices.  May also be a dict' + ' with key `hlines` specifying a price or sequence of prices, plus' + ' one or more of the following keys: `colors`, `linestyle`,' + ' `linewidths`, `alpha`.', 'Validator': _bypass_kwarg_validation}, 'vlines': {'Default': None, 'Description': 'Draw one or more VERTICAL LINES across entire plot, by' + ' specifying a date[time], or sequence of date[time].  May also' + ' be a dict with key `vlines` specifying a date[time] or sequence' + ' of date[time], plus one or more of the following keys:' + ' `colors`, `linestyle`, `linewidths`, `alpha`.', 'Validator': _bypass_kwarg_validation}, 'alines': {'Default': None, 'Description': 'Draw one or more ARBITRARY LINES anywhere on the plot, by' + ' specifying a sequence of two or more date/price pairs, or by' + ' specifying a sequence of sequences of two or more date/price pairs.' + ' May also be a dict with key `alines` (as date/price pairs described above),' + ' plus one or more of the following keys:' + ' `colors`, `linestyle`, `linewidths`, `alpha`.', 'Validator': _bypass_kwarg_validation}, 'tlines': {'Default': None, 'Description': 'Draw one or more TREND LINES by specifying one or more pairs of date[times]' + ' between which each trend line should be drawn.  May also be a dict with key' + ' `tlines` as just described, plus one or more of the following keys:' + ' `colors`, `linestyle`, `linewidths`, `alpha`, `tline_use`,`tline_method`.', 'Validator': _bypass_kwarg_validation}, 'colors': {'Default': None, 'Description': 'Color of [hvat]lines (or sequence of colors, if each line to be a different color)', 'Validator': lambda value: value is None or mcolors.is_color_like(value) or (isinstance(value, (list, tuple)) and all([mcolors.is_color_like(v) for v in value]))}, 'linestyle': {'Default': '-', 'Description': 'line style of [hvat]lines (or sequence of line styles, if each line to have a different linestyle)', 'Validator': lambda value: value is None or value in valid_linestyles or all([v in valid_linestyles for v in value])}, 'linewidths': {'Default': None, 'Description': 'line width of [hvat]lines (or sequence of line widths, if each line to have a different width)', 'Validator': lambda value: value is None or isinstance(value, (float, int)) or all([isinstance(v, (float, int)) for v in value])}, 'alpha': {'Default': 1.0, 'Description': 'Opacity of [hvat]lines (or sequence of opacities,' + 'if each line is to have a different opacity)' + 'float from 0.0 to 1.0 ' + ' (1.0 means fully opaque; 0.0 means transparent.', 'Validator': lambda value: isinstance(value, (float, int)) or all([isinstance(v, (float, int)) for v in value])}, 'tline_use': {'Default': 'close', 'Description': 'value to use for TREND LINE ("open","high","low","close") or sequence of' + ' any combination of "open", "high", "low", "close" to use a average of the' + ' specified values to determine the trend line.', 'Validator': lambda value: isinstance(value, str) or (isinstance(value, (list, tuple)) and all([isinstance(v, str) for v in value]))}, 'tline_method': {'Default': 'point-to-point', 'Description': 'method for TREND LINE determination: "point-to-point" or "least-squares"', 'Validator': lambda value: value in ['point-to-point', 'least-squares']}}
    _validate_vkwargs_dict(vkwargs)
    return vkwargs