import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _vlines_validator(value):
    """Validate `vlines` kwarg value:  must be "datelike" or sequence of "datelike"
    """
    if isinstance(value, dict):
        if 'vlines' in value:
            value = value['vlines']
        else:
            return False
    if _is_datelike(value):
        return True
    if not isinstance(value, (list, tuple)):
        return False
    if not all([_is_datelike(v) for v in value]):
        return False
    return True