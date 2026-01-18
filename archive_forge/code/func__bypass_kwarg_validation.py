import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _bypass_kwarg_validation(value):
    """ For some kwargs, we either don't know enough, or
        the validation is too complex to make it worth while,
        so we bypass kwarg validation.  If the kwarg is 
        invalid, then eventually an exception will be 
        raised at the time the kwarg value is actually used.
    """
    return True