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
@classmethod
def _from_args(cls, args, is_commutative=None):
    """Create new instance with already-processed args.
        If the args are not in canonical order, then a non-canonical
        result will be returned, so use with caution. The order of
        args may change if the sign of the args is changed."""
    if len(args) == 0:
        return cls.identity
    elif len(args) == 1:
        return args[0]
    obj = super().__new__(cls, *args)
    if is_commutative is None:
        is_commutative = fuzzy_and((a.is_commutative for a in args))
    obj.is_commutative = is_commutative
    return obj