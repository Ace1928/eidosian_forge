import logging
from typing import Dict, Optional, Tuple, Type
import sympy
from torch.utils._sympy.functions import FloorDiv
def _try_isolate_lhs(expr: sympy.Basic, thing: sympy.Basic, floordiv_inequality: bool) -> sympy.Basic:
    e = expr
    op = type(expr)
    if isinstance(e, sympy.Rel):
        lhs_not_thing = sum([a for a in e.lhs.args if not a.has(thing)]) if isinstance(e.lhs, sympy.Add) else 0
        e = op(expr.lhs - lhs_not_thing, expr.rhs - lhs_not_thing)
    if isinstance(e, sympy.Rel) and isinstance(e.lhs, sympy.Mul):
        lhs, rhs = e.args
        other = sympy.Mul(*[a for a in lhs.args if not a.has(thing)])
        if not (isinstance(e, INEQUALITY_TYPES) and other.is_negative is None):
            lhs = lhs / other
            rhs = rhs / other
            if isinstance(e, INEQUALITY_TYPES) and other.is_negative:
                op = mirror_rel_op(op)
            assert op is not None
            e = op(lhs, rhs)
    if floordiv_inequality and isinstance(e, sympy.Rel) and isinstance(e.lhs, FloorDiv) and e.lhs.divisor.is_positive and e.rhs.is_integer:
        if isinstance(expr, sympy.Eq):
            numerator, denominator = e.lhs.args
            return sympy.And(sympy.Ge(numerator, e.rhs * denominator), sympy.Lt(numerator, (e.rhs + 1) * denominator))
        if isinstance(expr, sympy.Ne):
            numerator, denominator = e.lhs.args
            return sympy.Or(sympy.Lt(numerator, e.rhs * denominator), sympy.Ge(numerator, (e.rhs + 1) * denominator))
        if isinstance(expr, (sympy.Gt, sympy.Ge)):
            quotient = e.rhs if isinstance(expr, sympy.Ge) else e.rhs + 1
            return sympy.Ge(e.lhs.args[0], quotient * e.lhs.args[1])
        if isinstance(expr, (sympy.Lt, sympy.Le)):
            quotient = e.rhs if isinstance(expr, sympy.Lt) else e.rhs + 1
            return sympy.Lt(e.lhs.args[0], quotient * e.lhs.args[1])
    return e