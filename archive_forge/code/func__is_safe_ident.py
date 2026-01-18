from __future__ import annotations
from typing import Any
import builtins
import inspect
import keyword
import textwrap
import linecache
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
from sympy.utilities.misc import filldedent
@classmethod
def _is_safe_ident(cls, ident):
    return isinstance(ident, str) and ident.isidentifier() and (not keyword.iskeyword(ident))