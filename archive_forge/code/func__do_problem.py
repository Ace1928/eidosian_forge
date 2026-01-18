import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def _do_problem(self, problem, integrator, method='adams'):

    def f(t, z):
        return problem.f(z, t)
    jac = None
    if hasattr(problem, 'jac'):

        def jac(t, z):
            return problem.jac(z, t)
    integrator_params = {}
    if problem.lband is not None or problem.uband is not None:
        integrator_params['uband'] = problem.uband
        integrator_params['lband'] = problem.lband
    ig = self.ode_class(f, jac)
    ig.set_integrator(integrator, atol=problem.atol / 10, rtol=problem.rtol / 10, method=method, **integrator_params)
    ig.set_initial_value(problem.z0, t=0.0)
    z = ig.integrate(problem.stop_t)
    assert_array_equal(z, ig.y)
    assert_(ig.successful(), (problem, method))
    assert_(ig.get_return_code() > 0, (problem, method))
    assert_(problem.verify(array([z]), problem.stop_t), (problem, method))