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
class BaseLayoutHierarchyType(BasePlotlyType):
    """
    Base class for all types in the layout hierarchy
    """

    @property
    def _parent_path_str(self):
        pass

    def __init__(self, plotly_name, **kwargs):
        super(BaseLayoutHierarchyType, self).__init__(plotly_name, **kwargs)

    def _send_prop_set(self, prop_path_str, val):
        if self.parent:
            self.parent._relayout_child(self, prop_path_str, val)