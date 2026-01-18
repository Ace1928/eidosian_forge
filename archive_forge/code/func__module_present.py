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
def _module_present(modname, modlist):
    if modname in modlist:
        return True
    for m in modlist:
        if hasattr(m, '__name__') and m.__name__ == modname:
            return True
    return False