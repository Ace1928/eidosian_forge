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
def Lambda(vs, body):
    """Create a Z3 lambda expression.

    >>> f = Function('f', IntSort(), IntSort(), IntSort())
    >>> mem0 = Array('mem0', IntSort(), IntSort())
    >>> lo, hi, e, i = Ints('lo hi e i')
    >>> mem1 = Lambda([i], If(And(lo <= i, i <= hi), e, mem0[i]))
    >>> mem1
    Lambda(i, If(And(lo <= i, i <= hi), e, mem0[i]))
    """
    ctx = body.ctx
    if is_app(vs):
        vs = [vs]
    num_vars = len(vs)
    _vs = (Ast * num_vars)()
    for i in range(num_vars):
        _vs[i] = vs[i].as_ast()
    return QuantifierRef(Z3_mk_lambda_const(ctx.ref(), num_vars, _vs, body.as_ast()), ctx)