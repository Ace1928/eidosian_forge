import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc
def _fix_invalid_lineno_ranges(astree: ast.AST):
    """Inplace fixes invalid lineno ranges.
    """
    ast.fix_missing_locations(astree)
    _EraseInvalidLineRanges().visit(astree)
    ast.fix_missing_locations(astree)