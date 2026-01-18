import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
def _test_Lambdify_scalar_vector_matrix(Lambdify):
    if not have_numpy:
        return
    args = x, y = se.symbols('x y')
    vec = se.DenseMatrix([x + y, x * y])
    jac = vec.jacobian(se.DenseMatrix(args))
    f = Lambdify(args, x ** y, vec, jac)
    assert f.n_exprs == 3
    s, v, m = f([2, 3])
    assert s == 2 ** 3
    assert np.allclose(v, [[2 + 3], [2 * 3]])
    assert np.allclose(m, [[1, 1], [3, 2]])
    for inp in [[2, 3, 5, 7], np.array([[2, 3], [5, 7]])]:
        s2, v2, m2 = f(inp)
        assert np.allclose(s2, [2 ** 3, 5 ** 7])
        assert np.allclose(v2, [[[2 + 3], [2 * 3]], [[5 + 7], [5 * 7]]])
        assert np.allclose(m2, [[[1, 1], [3, 2]], [[1, 1], [7, 5]]])