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
def _raise_invalid_rows_cols(name, n, invalid):
    rows_err_msg = '\n        If specified, the {name} parameter must be a list or tuple of integers\n        of length {n} (The number of traces being added)\n\n        Received: {invalid}\n        '.format(name=name, n=n, invalid=invalid)
    raise ValueError(rows_err_msg)