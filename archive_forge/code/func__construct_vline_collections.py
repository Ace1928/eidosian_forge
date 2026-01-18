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
def _construct_vline_collections(vlines, dtix, miny, maxy):
    """Construct vertical lines collection
    Parameters
    ----------
    vlines : sequence
        sequence of dates or datetimes at which to draw vertical lines
        dates/datetimes may be (a) pandas.to_datetime parseable string,
                               (b) pandas Timestamp
                               (c) python datetime.datetime or datetime.date

    vlines may also be a dict, containing
    the following keys:

        'vlines'     : the same as defined above: sequence of dates/datetimes
        'colors'     : colors for the above vlines
        'linestyle'  : line types for the above vlines
        'linewidths' : line widths for the above vlines

    dtix:  date index for the x-axis, used for converting the dates when
           x-values are 'evenly spaced integers' (as when skipping non-trading days)

    miny : minimum y-value for the vertical line

    maxy : maximum y-value for the vertical line

    Returns
    -------
    ret : list
        lines collections
    """
    if vlines is None:
        return None
    if isinstance(vlines, dict):
        vconfig = _process_kwargs(vlines, _valid_lines_kwargs())
        vlines = vconfig['vlines']
    else:
        vconfig = _process_kwargs({}, _valid_lines_kwargs())
    if not isinstance(vlines, (list, tuple)):
        vlines = [vlines]
    lines = []
    for val in vlines:
        lines.append([(val, miny), (val, maxy)])
    lines = _convert_segment_dates(lines, dtix)
    lw = vconfig['linewidths']
    co = vconfig['colors']
    ls = vconfig['linestyle']
    al = vconfig['alpha']
    lcollection = LineCollection(lines, colors=co, linewidths=lw, linestyles=ls, antialiaseds=(0,), alpha=al)
    return lcollection