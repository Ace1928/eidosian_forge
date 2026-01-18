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
def _in_batch_mode(self):
    """
        True if the object belongs to a figure that is currently in batch mode
        Returns
        -------
        bool
        """
    return self.parent and self.parent._in_batch_mode