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
def _construct_aline_collections(alines, dtix=None):
    """construct arbitrary line collections

    Parameters
    ----------
    alines : sequence
        sequences of segments, which are sequences of lines,
        which are sequences of two or more points ( date[time], price ) or (x,y)

        date[time] may be (a) pandas.to_datetime parseable string,
                          (b) pandas Timestamp, or
                          (c) python datetime.datetime or datetime.date

    alines may also be a dict, containing
    the following keys:

        'alines'     : the same as defined above: sequence of price, or dates, or segments
        'colors'     : colors for the above alines
        'linestyle'  : line types for the above alines
        'linewidths' : line widths for the above alines

    dtix:  date index for the x-axis, used for converting the dates when
           x-values are 'evenly spaced integers' (as when skipping non-trading days)

    Returns
    -------
    ret : list
        lines collections
    """
    if alines is None:
        return None
    if isinstance(alines, dict):
        aconfig = _process_kwargs(alines, _valid_lines_kwargs())
        alines = aconfig['alines']
    else:
        aconfig = _process_kwargs({}, _valid_lines_kwargs())
    alines = _alines_validator(alines, returnStandardizedValue=True)
    if alines is None:
        raise ValueError('Unable to standardize alines value: ' + str(alines))
    alines = _convert_segment_dates(alines, dtix)
    lw = aconfig['linewidths']
    co = aconfig['colors']
    ls = aconfig['linestyle']
    al = aconfig['alpha']
    lcollection = LineCollection(alines, colors=co, linewidths=lw, linestyles=ls, antialiaseds=(0,), alpha=al)
    return lcollection