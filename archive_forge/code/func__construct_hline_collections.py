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
def _construct_hline_collections(hlines, minx, maxx):
    """Construct horizontal lines collection

    Parameters
    ----------
    hlines : sequence
        sequence of [price] values at which to draw horizontal lines

    hlines may also be a dict, containing
    the following keys:

        'hlines'     : the same as defined above: sequence of price, or dates, or segments
        'colors'     : colors for the above hlines
        'linestyle'  : line types for the above hlines
        'linewidths' : line widths for the above hlines

    minx : the minimum value for x for the horizontal line, already converted to `xdates` format
    maxx : the maximum value for x for the horizontal line, already converted to `xdates` format

    Returns
    -------
    ret : list
        lines collections
    """
    if hlines is None:
        return None
    if isinstance(hlines, dict):
        hconfig = _process_kwargs(hlines, _valid_lines_kwargs())
        hlines = hconfig['hlines']
    else:
        hconfig = _process_kwargs({}, _valid_lines_kwargs())
    lines = []
    if not isinstance(hlines, (list, tuple)):
        hlines = [hlines]
    for val in hlines:
        lines.append([(minx, val), (maxx, val)])
    lw = hconfig['linewidths']
    co = hconfig['colors']
    ls = hconfig['linestyle']
    al = hconfig['alpha']
    lcollection = LineCollection(lines, colors=co, linewidths=lw, linestyles=ls, antialiaseds=(0,), alpha=al)
    return lcollection