from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def _eval_unary_minus(evaluator, tree):
    if tree.args[0].type == 'ZERO':
        return IntermediateExpr(True, tree.origin, False, [])
    elif tree.args[0].type == 'ONE':
        return IntermediateExpr(False, None, True, [])
    else:
        raise PatsyError('Unary minus can only be applied to 1 or 0', tree)