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
class ValidatorCompat(object):

    def __getitem__(self, item):
        return obj._get_validator(item)

    def __contains__(self, item):
        return obj.__contains__(item)

    def __iter__(self):
        return iter(obj)

    def items(self):
        return [(k, self[k]) for k in self]