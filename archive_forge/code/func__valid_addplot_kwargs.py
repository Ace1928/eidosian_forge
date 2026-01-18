import matplotlib.dates  as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.axes   as mpl_axes
import matplotlib.figure as mpl_fig
import pandas as pd
import numpy  as np
import copy
import io
import os
import math
import warnings
import statistics as stat
from itertools import cycle
from mplfinance._utils import _construct_aline_collections
from mplfinance._utils import _construct_hline_collections
from mplfinance._utils import _construct_vline_collections
from mplfinance._utils import _construct_tline_collections
from mplfinance._utils import _construct_mpf_collections
from mplfinance._utils import _construct_pnf_scatter
from mplfinance._widths import _determine_width_config
from mplfinance._utils import _updown_colors
from mplfinance._utils import IntegerIndexDateTimeFormatter
from mplfinance._utils import _mscatter
from mplfinance._utils import _check_and_convert_xlim_configuration
from mplfinance import _styles
from mplfinance._arg_validators import _check_and_prepare_data, _mav_validator, _label_validator
from mplfinance._arg_validators import _get_valid_plot_types, _fill_between_validator
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _kwarg_not_implemented, _bypass_kwarg_validation
from mplfinance._arg_validators import _hlines_validator, _vlines_validator
from mplfinance._arg_validators import _alines_validator, _tlines_validator
from mplfinance._arg_validators import _scale_padding_validator, _yscale_validator
from mplfinance._arg_validators import _valid_panel_id, _check_for_external_axes
from mplfinance._arg_validators import _xlim_validator, _mco_validator, _is_marketcolor_object
from mplfinance._panels import _build_panels
from mplfinance._panels import _set_ticks_on_bottom_panel_only
from mplfinance._helpers import _determine_format_string
from mplfinance._helpers import _list_of_dict
from mplfinance._helpers import _num_or_seq_of_num
from mplfinance._helpers import _adjust_color_brightness
def _valid_addplot_kwargs():
    valid_linestyles = ('-', 'solid', '--', 'dashed', '-.', 'dashdot', ':', 'dotted', None, ' ', '')
    valid_types = ('line', 'scatter', 'bar', 'ohlc', 'candle', 'step')
    valid_stepwheres = ('pre', 'post', 'mid')
    valid_edgecolors = ('face', 'none', None)
    vkwargs = {'scatter': {'Default': False, 'Description': "Deprecated.  (Use kwarg `type='scatter' instead.", 'Validator': lambda value: isinstance(value, bool)}, 'type': {'Default': 'line', 'Description': 'addplot type: "line","scatter","bar", "ohlc", "candle","step"', 'Validator': lambda value: value in valid_types}, 'mav': {'Default': None, 'Description': 'Moving Average window size(s); (int or tuple of ints)', 'Validator': _mav_validator}, 'panel': {'Default': 0, 'Description': 'Panel (int 0-31) to use for this addplot', 'Validator': lambda value: _valid_panel_id(value)}, 'marker': {'Default': 'o', 'Description': "marker for `type='scatter'` plot", 'Validator': lambda value: _bypass_kwarg_validation(value)}, 'markersize': {'Default': 18, 'Description': 'size of marker for `type="scatter"`; default=18', 'Validator': lambda value: isinstance(value, (int, float, pd.Series, np.ndarray))}, 'color': {'Default': None, 'Description': 'color (or sequence of colors) of line(s), scatter marker(s), or bar(s).', 'Validator': lambda value: mcolors.is_color_like(value) or (isinstance(value, (list, tuple, np.ndarray)) and all([mcolors.is_color_like(v) for v in value]))}, 'linestyle': {'Default': None, 'Description': 'line style for `type=line` (' + str(valid_linestyles) + ')', 'Validator': lambda value: value in valid_linestyles}, 'linewidths': {'Default': None, 'Description': 'edge widths of scatter markers', 'Validator': lambda value: isinstance(value, (int, float))}, 'edgecolors': {'Default': None, 'Description': 'edgecolors of scatter markers', 'Validator': lambda value: mcolors.is_color_like(value) or value in valid_edgecolors}, 'width': {'Default': None, 'Description': 'width of bar or line for `type="bar"` or `type="line"', 'Validator': lambda value: isinstance(value, (int, float)) or all([isinstance(v, (int, float)) for v in value])}, 'bottom': {'Default': 0, 'Description': 'bottom value for `type=bar` bars. Default=0', 'Validator': lambda value: isinstance(value, (int, float)) or all([isinstance(v, (int, float)) for v in value])}, 'alpha': {'Default': 1, 'Description': 'opacity for 0.0 (transparent) to 1.0 (opaque)', 'Validator': lambda value: isinstance(value, (int, float)) or all([isinstance(v, (int, float)) for v in value])}, 'secondary_y': {'Default': 'auto', 'Description': "True|False|'auto' place the additional plot data on a" + " secondary y-axis.  'auto' compares the magnitude or the" + ' addplot data, to data already on the axis, and if it appears' + ' they are of different magnitudes, then it uses a secondary y-axis.' + " True or False always override 'auto'.", 'Validator': lambda value: isinstance(value, bool) or value == 'auto'}, 'y_on_right': {'Default': None, 'Description': 'True|False put y-axis tick labels on the right, for this addplot' + ' regardless of what the mplfinance style says to to.', 'Validator': lambda value: isinstance(value, bool)}, 'ylabel': {'Default': None, 'Description': 'label for y-axis (for this addplot)', 'Validator': lambda value: isinstance(value, str)}, 'ylim': {'Default': None, 'Description': 'Limits for addplot y-axis as tuple (min,max), i.e. (bottom,top)', 'Validator': lambda value: isinstance(value, (list, tuple)) and len(value) == 2 and all([isinstance(v, (int, float)) for v in value])}, 'title': {'Default': None, 'Description': 'Axes Title (subplot title) for this addplot.', 'Validator': lambda value: isinstance(value, str)}, 'ax': {'Default': None, 'Description': 'Matplotlib Axes object on which to plot this addplot', 'Validator': lambda value: isinstance(value, mpl_axes.Axes)}, 'yscale': {'Default': None, 'Description': 'addplot y-axis scale: "linear", "log", "symlog", or "logit"', 'Validator': lambda value: _yscale_validator(value)}, 'stepwhere': {'Default': 'pre', 'Description': "'pre','post', or 'mid': where to place step relative" + " to data for `type='step'`", 'Validator': lambda value: value in valid_stepwheres}, 'marketcolors': {'Default': None, 'Description': 'marketcolors for this addplot (instead of the mplfinance' + " style's marketcolors).  For addplot `type='ohlc'`" + " and type='candle'", 'Validator': lambda value: _is_marketcolor_object(value)}, 'fill_between': {'Default': None, 'Description': ' fill region', 'Validator': _fill_between_validator}, 'label': {'Default': None, 'Description': 'Label for the added plot. One per added plot.', 'Validator': _label_validator}}
    _validate_vkwargs_dict(vkwargs)
    return vkwargs