from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def DeclareTypeVar(name, ctx=None):
    """Create a new type variable named `name`.

    If `ctx=None`, then the new sort is declared in the global Z3Py context.

    """
    ctx = _get_ctx(ctx)
    return TypeVarRef(Z3_mk_type_variable(ctx.ref(), to_symbol(name, ctx)), ctx)