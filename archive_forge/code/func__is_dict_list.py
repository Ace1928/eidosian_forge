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
def _is_dict_list(v):
    """
        Return true of the input object is a list of dicts
        """
    return isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)