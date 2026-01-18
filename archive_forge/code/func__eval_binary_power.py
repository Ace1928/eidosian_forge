from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def _eval_binary_power(evaluator, tree):
    left_expr = evaluator.eval(tree.args[0])
    _check_interactable(left_expr)
    power = -1
    if tree.args[1].type in ('ONE', 'NUMBER'):
        expr = tree.args[1].token.extra
        try:
            power = int(expr)
        except ValueError:
            pass
    if power < 1:
        raise PatsyError("'**' requires a positive integer", tree.args[1])
    all_terms = left_expr.terms
    big_expr = left_expr
    power = min(len(left_expr.terms), power)
    for i in range(1, power):
        big_expr = _interaction(left_expr, big_expr)
        all_terms = all_terms + big_expr.terms
    return IntermediateExpr(False, None, False, all_terms)