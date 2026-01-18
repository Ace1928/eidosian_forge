from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def _to_list_of_floats(s):
    return [_to_float(val) for val in s.split()]