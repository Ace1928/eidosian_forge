import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class TestOdeint:

    def _do_problem(self, problem):
        t = arange(0.0, problem.stop_t, 0.05)
        z, infodict = odeint(problem.f, problem.z0, t, full_output=True)
        assert_(problem.verify(z, t))
        z, infodict = odeint(lambda t, y: problem.f(y, t), problem.z0, t, full_output=True, tfirst=True)
        assert_(problem.verify(z, t))
        if hasattr(problem, 'jac'):
            z, infodict = odeint(problem.f, problem.z0, t, Dfun=problem.jac, full_output=True)
            assert_(problem.verify(z, t))
            z, infodict = odeint(lambda t, y: problem.f(y, t), problem.z0, t, Dfun=lambda t, y: problem.jac(y, t), full_output=True, tfirst=True)
            assert_(problem.verify(z, t))

    def test_odeint(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem)