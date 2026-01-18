from collections import OrderedDict
import dateutil.parser
import itertools
import logging
import numpy as np
from matplotlib import _api, ticker, units
@staticmethod
def _str_is_convertible(val):
    """
        Helper method to check whether a string can be parsed as float or date.
        """
    try:
        float(val)
    except ValueError:
        try:
            dateutil.parser.parse(val)
        except (ValueError, TypeError):
            return False
    return True