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
def _get_subplot_coordinates(self):
    """
        Returns an iterator over (row,col) pairs representing all the possible
        subplot coordinates.
        """
    return itertools.product(*self._get_subplot_rows_columns())