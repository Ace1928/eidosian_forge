import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
@staticmethod
def _is_key_path_compatible(key_path_str, plotly_obj):
    """
        Return whether the specifieid key path string is compatible with
        the specified plotly object for the purpose of relayout/restyle
        operation
        """
    key_path_tuple = BaseFigure._str_to_dict_path(key_path_str)
    if isinstance(key_path_tuple[-1], int):
        key_path_tuple = key_path_tuple[:-1]
    return key_path_tuple in plotly_obj