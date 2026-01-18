import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def _is_col_list(columns, arg):
    """Returns True if arg looks like it's a list of columns or references to columns
    in df_input, and False otherwise (in which case it's assumed to be a single column
    or reference to a column).
    """
    if arg is None or isinstance(arg, str) or isinstance(arg, int):
        return False
    if isinstance(arg, pd.MultiIndex):
        return False
    try:
        iter(arg)
    except TypeError:
        return False
    for c in arg:
        if isinstance(c, str) or isinstance(c, int):
            if columns is None or c not in columns:
                return False
        else:
            try:
                iter(c)
            except TypeError:
                return False
    return True