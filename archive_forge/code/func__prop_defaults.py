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
@property
def _prop_defaults(self):
    """
        Return default properties dict

        Returns
        -------
        dict
        """
    if self.parent is None:
        return None
    else:
        return self.parent._get_child_prop_defaults(self)