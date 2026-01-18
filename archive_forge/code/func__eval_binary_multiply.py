from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def _eval_binary_multiply(self, tree):
    left = self.eval(tree.args[0])
    right = self.eval(tree.args[1])
    if self.is_constant(left):
        return left[-1] * right
    elif self.is_constant(right):
        return left * right[-1]
    else:
        raise PatsyError("Can't multiply one variable by another in a linear constraint", tree)