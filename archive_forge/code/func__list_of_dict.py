import datetime
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import numpy as np
def _list_of_dict(x):
    """
    Return True if x is a list of dict's
    """
    return isinstance(x, list) and all([isinstance(item, dict) for item in x])