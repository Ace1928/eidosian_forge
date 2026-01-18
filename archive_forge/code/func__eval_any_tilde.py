from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def _eval_any_tilde(evaluator, tree):
    exprs = [evaluator.eval(arg) for arg in tree.args]
    if len(exprs) == 1:
        exprs.insert(0, IntermediateExpr(False, None, True, []))
    assert len(exprs) == 2
    return ModelDesc(_maybe_add_intercept(exprs[0].intercept, exprs[0].terms), _maybe_add_intercept(not exprs[1].intercept_removed, exprs[1].terms))