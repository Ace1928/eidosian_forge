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
@classmethod
def _exec_constructor_postprocessors(cls, obj):
    clsname = obj.__class__.__name__
    postprocessors = defaultdict(list)
    for i in obj.args:
        try:
            postprocessor_mappings = (Basic._constructor_postprocessor_mapping[cls].items() for cls in type(i).mro() if cls in Basic._constructor_postprocessor_mapping)
            for k, v in chain.from_iterable(postprocessor_mappings):
                postprocessors[k].extend([j for j in v if j not in postprocessors[k]])
        except TypeError:
            pass
    for f in postprocessors.get(clsname, []):
        obj = f(obj)
    return obj