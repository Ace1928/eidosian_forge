import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
def _check_action(fun, args, action):
    if action == 'warn':
        with pytest.warns(sc.SpecialFunctionWarning):
            fun(*args)
    elif action == 'raise':
        with assert_raises(sc.SpecialFunctionError):
            fun(*args)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fun(*args)