from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def _eval_variable(self, tree):
    var = tree.token.extra
    coefs = np.zeros((self._N + 1,), dtype=float)
    coefs[self._variable_names.index(var)] = 1
    return coefs