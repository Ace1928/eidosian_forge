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
def _dispatch_on_deselect(self, points):
    """
        Dispatch points info to deselection callbacks
        """
    if 'selectedpoints' in self:
        self.selectedpoints = None
    for callback in self._deselect_callbacks:
        callback(self, points)