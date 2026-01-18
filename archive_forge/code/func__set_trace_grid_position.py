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
def _set_trace_grid_position(self, trace, row, col, secondary_y=False):
    from plotly._subplots import _set_trace_grid_reference
    grid_ref = self._validate_get_grid_ref()
    return _set_trace_grid_reference(trace, self.layout, grid_ref, row, col, secondary_y)