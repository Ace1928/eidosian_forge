import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode
def _solve_linear_sys(a, y0, tend=1, dt=0.1, solver=None, method='bdf', use_jac=True, with_jacobian=False, banded=False):
    """Use scipy.integrate.ode to solve a linear system of ODEs.

    a : square ndarray
        Matrix of the linear system to be solved.
    y0 : ndarray
        Initial condition
    tend : float
        Stop time.
    dt : float
        Step size of the output.
    solver : str
        If not None, this must be "vode", "lsoda" or "zvode".
    method : str
        Either "bdf" or "adams".
    use_jac : bool
        Determines if the jacobian function is passed to ode().
    with_jacobian : bool
        Passed to ode.set_integrator().
    banded : bool
        Determines whether a banded or full jacobian is used.
        If `banded` is True, `lband` and `uband` are determined by the
        values in `a`.
    """
    if banded:
        lband, uband = _band_count(a)
    else:
        lband = None
        uband = None
    if use_jac:
        if banded:
            r = ode(_linear_func, _linear_banded_jac)
        else:
            r = ode(_linear_func, _linear_jac)
    else:
        r = ode(_linear_func)
    if solver is None:
        if np.iscomplexobj(a):
            solver = 'zvode'
        else:
            solver = 'vode'
    r.set_integrator(solver, with_jacobian=with_jacobian, method=method, lband=lband, uband=uband, rtol=1e-09, atol=1e-10)
    t0 = 0
    r.set_initial_value(y0, t0)
    r.set_f_params(a)
    r.set_jac_params(a)
    t = [t0]
    y = [y0]
    while r.successful() and r.t < tend:
        r.integrate(r.t + dt)
        t.append(r.t)
        y.append(r.y)
    t = np.array(t)
    y = np.array(y)
    return (t, y)