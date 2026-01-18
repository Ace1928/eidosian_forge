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
@staticmethod
def _recursive_call(expr_to_call, on_args):
    """Helper for rcall method."""
    from .symbol import Symbol

    def the_call_method_is_overridden(expr):
        for cls in getmro(type(expr)):
            if '__call__' in cls.__dict__:
                return cls != Basic
    if callable(expr_to_call) and the_call_method_is_overridden(expr_to_call):
        if isinstance(expr_to_call, Symbol):
            return expr_to_call
        else:
            return expr_to_call(*on_args)
    elif expr_to_call.args:
        args = [Basic._recursive_call(sub, on_args) for sub in expr_to_call.args]
        return type(expr_to_call)(*args)
    else:
        return expr_to_call