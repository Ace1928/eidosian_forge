from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def _check_lincon(input, varnames, coefs, constants):
    try:
        from numpy.testing import assert_equal
    except ImportError:
        from numpy.testing.utils import assert_equal
    got = linear_constraint(input, varnames)
    print('got', got)
    expected = LinearConstraint(varnames, coefs, constants)
    print('expected', expected)
    assert_equal(got.variable_names, expected.variable_names)
    assert_equal(got.coefs, expected.coefs)
    assert_equal(got.constants, expected.constants)
    assert_equal(got.coefs.dtype, np.dtype(float))
    assert_equal(got.constants.dtype, np.dtype(float))