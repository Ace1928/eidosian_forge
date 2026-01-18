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
def _index_is(iterable, val):
    """
        Return the index of a value in an iterable using object identity
        (not object equality as is the case for list.index)

        """
    index_list = [i for i, curr_val in enumerate(iterable) if curr_val is val]
    if not index_list:
        raise ValueError('Invalid value')
    return index_list[0]