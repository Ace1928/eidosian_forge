from __future__ import annotations
from operator import attrgetter
from collections import defaultdict
from sympy.utilities.exceptions import sympy_deprecation_warning
from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import (Dispatcher,
@staticmethod
def _compare_pretty(a, b):
    return (str(a) > str(b)) - (str(a) < str(b))