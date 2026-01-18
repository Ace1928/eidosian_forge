import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _mco_validator(value):
    if isinstance(value, dict):
        if 'colors' not in value:
            raise ValueError('`marketcolor_overrides` as dict must contain `colors` key.')
        colors = value['colors']
    else:
        colors = value
    if not isinstance(colors, (list, tuple, np.ndarray)):
        return False
    return all([c is None or _mpf_is_color_like(c) or _is_marketcolor_object(c) for c in colors])