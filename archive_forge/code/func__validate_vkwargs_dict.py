import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _validate_vkwargs_dict(vkwargs):
    """
    Check that we didn't make a typo in any of the things
    that should be the same for all vkwargs dict items:

    {kwarg : {'Default': ... , 'Description': ... , 'Validator': ...} }
    """
    for key, value in vkwargs.items():
        if len(value) != 3:
            raise ValueError('Items != 3 in valid kwarg table, for kwarg "' + key + '"')
        if 'Default' not in value:
            raise ValueError('Missing "Default" value for kwarg "' + key + '"')
        if 'Description' not in value:
            raise ValueError('Missing "Description" value for kwarg "' + key + '"')
        if 'Validator' not in value:
            raise ValueError('Missing "Validator" function for kwarg "' + key + '"')