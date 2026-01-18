import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
class _Undefined:
    """
    A sentinel value for undefined variable created by Expr.undef.
    """

    def __repr__(self):
        return '<undefined>'