import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _valid_mav(value, is_period=True):
    if not isinstance(value, (tuple, list, int)):
        return False
    if isinstance(value, int):
        return value >= 2 or not is_period
    for num in value:
        if not isinstance(num, int) or (is_period and num < 2):
            return False
    return True