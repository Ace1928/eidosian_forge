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
def _valid_pnf_kwargs():
    """
    Construct and return the "valid pnf kwargs table" for the mplfinance.plot(type='pnf')
    function. A valid kwargs table is a `dict` of `dict`s. The keys of the outer dict are
    the valid key-words for the function.  The value for each key is a dict containing 3
    specific keys: "Default", "Description" and "Validator" with the following values:
        "Default"      - The default value for the kwarg if none is specified.
        "Description"  - The description for the kwarg.
        "Validator"    - A function that takes the caller specified value for the kwarg,
                         and validates that it is the correct type, and (for kwargs with
                         a limited set of allowed values) may also validate that the
                         kwarg value is one of the allowed values.
    """

    def _box_size_validator(v):
        if isinstance(v, (float, int)):
            return True
        if v == 'atr':
            return True
        if isinstance(v, str) and v[-1:] == '%' and v[:-1].replace('.', '', 1).isdigit():
            return True
        return False
    vkwargs = {'box_size': {'Default': 'atr', 'Description': 'size of each box on y-axis (typically price).' + ' specify a number, or "atr" for average true range' + ' or a string containing a number and "%" for' + ' percent of the most recent close price.', 'Validator': lambda value: _box_size_validator(value)}, 'atr_length': {'Default': 'total', 'Description': 'number of periods for atr calculation (if box size is "atr")', 'Validator': lambda value: isinstance(value, int) or value == 'total'}, 'reversal': {'Default': 3, 'Description': 'number of boxes, in opposite direction, needed to reverse' + ' a trend (i.e. to start a new column).', 'Validator': lambda value: isinstance(value, int)}, 'method': {'Default': 'hilo', 'Description': 'pricing method:' + ' specify "hilo" to use High for X and Low for O' + ' or specify "open" or "close" to use only Open or only Close price.', 'Validator': lambda value: value in ['hilo', 'open', 'close']}, 'use_candle_colors': {'Default': False, 'Description': 'use same colors as candles for given style' + ' (instead of PNF colors derived from candle colors).', 'Validator': lambda value: isinstance(value, bool)}, 'scale_markers': {'Default': 1.0, 'Description': 'Scale PNF markers larger ( > 1.0) or smaller ( < 1.0)', 'Validator': lambda value: isinstance(value, (int, float))}, 'scale_right_padding': {'Default': 1.0, 'Description': 'Scale the amount of padding on the right side' + ' of the plot. (Padding helps the PnF remain square', 'Validator': lambda value: isinstance(value, (int, float))}}
    _validate_vkwargs_dict(vkwargs)
    return vkwargs