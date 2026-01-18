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
def _add_domain(ax_letter, new_axref):
    axref = ax_letter + 'ref'
    if axref in new_obj._props.keys() and 'domain' in new_obj[axref]:
        new_axref += ' domain'
    return new_axref