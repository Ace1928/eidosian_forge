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
def _split_and_chomp(s):
    if not len(s):
        return s
    s_split = split_multichar([s], list('_'))
    s_chomped = chomp_empty_strings(s_split, '_', reverse=True)
    return s_chomped