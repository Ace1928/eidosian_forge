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
def _normalize_trace_indexes(self, trace_indexes):
    """
        Input trace index specification and return list of the specified trace
        indexes

        Parameters
        ----------
        trace_indexes : None or int or list[int]

        Returns
        -------
        list[int]
        """
    if trace_indexes is None:
        trace_indexes = list(range(len(self.data)))
    if not isinstance(trace_indexes, (list, tuple)):
        trace_indexes = [trace_indexes]
    return list(trace_indexes)