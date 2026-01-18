import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
class TestNewToOldCobyla:
    method = 'cobyla'
    list_of_problems = [Elec(n_electrons=2), Elec(n_electrons=4)]

    @pytest.mark.slow
    def test_list_of_problems(self):
        for prob in self.list_of_problems:
            with suppress_warnings() as sup:
                sup.filter(UserWarning)
                truth = minimize(prob.fun, prob.x0, method='trust-constr', bounds=prob.bounds, constraints=prob.constr)
                result = minimize(prob.fun, prob.x0, method=self.method, bounds=prob.bounds, constraints=prob.constr)
            assert_allclose(result.fun, truth.fun, rtol=0.001)