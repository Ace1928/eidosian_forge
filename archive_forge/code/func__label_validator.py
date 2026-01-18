import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _label_validator(label_value):
    """ Validates the input of [legend] label for added plots.
    label_value may be a str or a sequence of str.
    """
    if isinstance(label_value, str):
        return True
    if isinstance(label_value, (list, tuple, np.ndarray)):
        if all([isinstance(v, str) for v in label_value]):
            return True
    return False