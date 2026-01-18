import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _scale_padding_validator(value):
    if isinstance(value, (int, float)):
        return True
    elif isinstance(value, dict):
        valid_keys = ('left', 'right', 'top', 'bottom')
        for key in value:
            if key not in valid_keys:
                raise ValueError('Invalid key "' + str(key) + '" found in `scale_padding` dict.')
            if not isinstance(value[key], (int, float)):
                raise ValueError('`scale_padding` dict contains non-number at key "' + str(key) + '"')
        return True
    else:
        raise ValueError('`scale_padding` kwarg must be a number, or dict of (left,right,top,bottom) numbers.')
    return False