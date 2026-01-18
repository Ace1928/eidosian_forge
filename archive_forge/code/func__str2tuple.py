import re
import copy
import json
import warnings
from .symbol import Symbol
def _str2tuple(string):
    """Convert shape string to list, internal use only.

    Parameters
    ----------
    string: str
        Shape string.

    Returns
    -------
    list of str
        Represents shape.
    """
    return re.findall('\\d+', string)