from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def _eval_binary_eq(self, tree):
    args = list(tree.args)
    constraints = []
    for i, arg in enumerate(args):
        if arg.type == '=':
            constraints.append(self.eval(arg, constraint=True))
            args[i] = arg.args[1 - i]
    left = self.eval(args[0])
    right = self.eval(args[1])
    coefs = left[:self._N] - right[:self._N]
    if np.all(coefs == 0):
        raise PatsyError('no variables appear in constraint', tree)
    constant = -left[-1] + right[-1]
    constraint = LinearConstraint(self._variable_names, coefs, constant)
    constraints.append(constraint)
    return LinearConstraint.combine(constraints)