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
def _validate_get_grid_ref(self):
    try:
        grid_ref = self._grid_ref
        if grid_ref is None:
            raise AttributeError('_grid_ref')
    except AttributeError:
        raise Exception('In order to reference traces by row and column, you must first use plotly.tools.make_subplots to create the figure with a subplot grid.')
    return grid_ref