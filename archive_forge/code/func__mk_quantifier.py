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
def _mk_quantifier(is_forall, vs, body, weight=1, qid='', skid='', patterns=[], no_patterns=[]):
    if z3_debug():
        _z3_assert(is_bool(body) or is_app(vs) or (len(vs) > 0 and is_app(vs[0])), 'Z3 expression expected')
        _z3_assert(is_const(vs) or (len(vs) > 0 and all([is_const(v) for v in vs])), 'Invalid bounded variable(s)')
        _z3_assert(all([is_pattern(a) or is_expr(a) for a in patterns]), 'Z3 patterns expected')
        _z3_assert(all([is_expr(p) for p in no_patterns]), 'no patterns are Z3 expressions')
    if is_app(vs):
        ctx = vs.ctx
        vs = [vs]
    else:
        ctx = vs[0].ctx
    if not is_expr(body):
        body = BoolVal(body, ctx)
    num_vars = len(vs)
    if num_vars == 0:
        return body
    _vs = (Ast * num_vars)()
    for i in range(num_vars):
        _vs[i] = vs[i].as_ast()
    patterns = [_to_pattern(p) for p in patterns]
    num_pats = len(patterns)
    _pats = (Pattern * num_pats)()
    for i in range(num_pats):
        _pats[i] = patterns[i].ast
    _no_pats, num_no_pats = _to_ast_array(no_patterns)
    qid = to_symbol(qid, ctx)
    skid = to_symbol(skid, ctx)
    return QuantifierRef(Z3_mk_quantifier_const_ex(ctx.ref(), is_forall, weight, qid, skid, num_vars, _vs, num_pats, _pats, num_no_pats, _no_pats, body.as_ast()), ctx)