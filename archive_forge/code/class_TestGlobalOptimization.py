import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
class TestGlobalOptimization:

    def test_optimize_result_attributes(self):

        def func(x):
            return x ** 2
        results = [optimize.basinhopping(func, x0=1), optimize.differential_evolution(func, [(-4, 4)]), optimize.shgo(func, [(-4, 4)]), optimize.dual_annealing(func, [(-4, 4)]), optimize.direct(func, [(-4, 4)])]
        for result in results:
            assert isinstance(result, optimize.OptimizeResult)
            assert hasattr(result, 'x')
            assert hasattr(result, 'success')
            assert hasattr(result, 'message')
            assert hasattr(result, 'fun')
            assert hasattr(result, 'nfev')
            assert hasattr(result, 'nit')