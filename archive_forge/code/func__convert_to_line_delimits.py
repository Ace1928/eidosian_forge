import copy
from collections import defaultdict
import numpy as np
from pandas import compat, DataFrame
def _convert_to_line_delimits(s):
    """Helper function that converts json lists to line delimited json."""
    if not s[0] == '[' and s[-1] == ']':
        return s
    s = s[1:-1]
    return s