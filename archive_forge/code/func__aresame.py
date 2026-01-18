from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from itertools import chain, zip_longest
from .assumptions import _prepare_class_assumptions
from .cache import cacheit
from .core import ordering_of_classes
from .sympify import _sympify, sympify, SympifyError, _external_converter
from .sorting import ordered
from .kind import Kind, UndefinedKind
from ._print_helpers import Printable
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, numbered_symbols
from sympy.utilities.misc import filldedent, func_name
from inspect import getmro
from .singleton import S
from .traversal import (preorder_traversal as _preorder_traversal,
def _aresame(a, b):
    """Return True if a and b are structurally the same, else False.

    Examples
    ========

    In SymPy (as in Python) two numbers compare the same if they
    have the same underlying base-2 representation even though
    they may not be the same type:

    >>> from sympy import S
    >>> 2.0 == S(2)
    True
    >>> 0.5 == S.Half
    True

    This routine was written to provide a query for such cases that
    would give false when the types do not match:

    >>> from sympy.core.basic import _aresame
    >>> _aresame(S(2.0), S(2))
    False

    """
    from .numbers import Number
    from .function import AppliedUndef, UndefinedFunction as UndefFunc
    if isinstance(a, Number) and isinstance(b, Number):
        return a == b and a.__class__ == b.__class__
    for i, j in zip_longest(_preorder_traversal(a), _preorder_traversal(b)):
        if i != j or type(i) != type(j):
            if isinstance(i, UndefFunc) and isinstance(j, UndefFunc) or (isinstance(i, AppliedUndef) and isinstance(j, AppliedUndef)):
                if i.class_key() != j.class_key():
                    return False
            else:
                return False
    return True