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
def _array_select(ar, arg):
    if isinstance(arg, tuple):
        args = [ar.sort().domain_n(i).cast(arg[i]) for i in range(len(arg))]
        _args, sz = _to_ast_array(args)
        return _to_expr_ref(Z3_mk_select_n(ar.ctx_ref(), ar.as_ast(), sz, _args), ar.ctx)
    arg = ar.sort().domain().cast(arg)
    return _to_expr_ref(Z3_mk_select(ar.ctx_ref(), ar.as_ast(), arg.as_ast()), ar.ctx)