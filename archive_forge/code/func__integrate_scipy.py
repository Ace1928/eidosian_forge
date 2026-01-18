from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _integrate_scipy(self, intern_xout, intern_y0, intern_p, atol=1e-08, rtol=1e-08, first_step=None, with_jacobian=None, force_predefined=False, name=None, **kwargs):
    """ Do not use directly (use ``integrate('scipy', ...)``).

        Uses `scipy.integrate.ode <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_

        Parameters
        ----------
        \\*args :
            See :meth:`integrate`.
        name : str (default: 'lsoda'/'dopri5' when jacobian is available/not)
            What integrator wrapped in scipy.integrate.ode to use.
        \\*\\*kwargs :
            Keyword arguments passed onto `set_integrator(...) <
        http://docs.scipy.org/doc/scipy/reference/generated/
        scipy.integrate.ode.set_integrator.html#scipy.integrate.ode.set_integrator>`_

        Returns
        -------
        See :meth:`integrate`.
        """
    from scipy.integrate import ode
    ny = intern_y0.shape[-1]
    nx = intern_xout.shape[-1]
    results = []
    for _xout, _y0, _p in zip(intern_xout, intern_y0, intern_p):
        if name is None:
            if self.j_cb is None:
                name = 'dopri5'
            else:
                name = 'lsoda'
        if with_jacobian is None:
            if name == 'lsoda':
                with_jacobian = True
            elif name in ('dop853', 'dopri5'):
                with_jacobian = False
            elif name == 'vode':
                with_jacobian = kwargs.get('method', 'adams') == 'bdf'

        def rhs(t, y, p=()):
            rhs.ncall += 1
            return self.f_cb(t, y, p)
        rhs.ncall = 0
        if self.j_cb is not None:

            def jac(t, y, p=()):
                jac.ncall += 1
                return self.j_cb(t, y, p)
            jac.ncall = 0
        r = ode(rhs, jac=jac if with_jacobian else None)
        if 'lband' in kwargs or 'uband' in kwargs or 'band' in kwargs:
            raise ValueError('lband and uband set locally (set `band` at initialization instead)')
        if self.band is not None:
            kwargs['lband'], kwargs['uband'] = self.band
        r.set_integrator(name, atol=atol, rtol=rtol, **kwargs)
        if len(_p) > 0:
            r.set_f_params(_p)
            r.set_jac_params(_p)
        r.set_initial_value(_y0, _xout[0])
        if nx == 2 and (not force_predefined):
            mode = 'adaptive'
            if name in ('vode', 'lsoda'):
                warnings.warn("'adaptive' mode with SciPy's integrator (vode/lsoda) may overshoot (itask=2)")
                warnings.warn("'adaptive' mode with SciPy's integrator is unreliable, consider using e.g. cvode")
                ysteps = [_y0]
                xsteps = [_xout[0]]
                while r.t < _xout[1]:
                    r.integrate(_xout[1], step=True)
                    if not r.successful():
                        raise RuntimeError('failed')
                    xsteps.append(r.t)
                    ysteps.append(r.y)
            else:
                xsteps, ysteps = ([], [])

                def solout(x, y):
                    xsteps.append(x)
                    ysteps.append(y)
                r.set_solout(solout)
                r.integrate(_xout[1])
                if not r.successful():
                    raise RuntimeError('failed')
            _yout = np.array(ysteps)
            _xout = np.array(xsteps)
        else:
            mode = 'predefined'
            _yout = np.empty((nx, ny))
            _yout[0, :] = _y0
            for idx in range(1, nx):
                r.integrate(_xout[idx])
                if not r.successful():
                    raise RuntimeError('failed')
                _yout[idx, :] = r.y
        info = {'internal_xout': _xout, 'internal_yout': _yout, 'internal_params': _p, 'success': r.successful(), 'nfev': rhs.ncall, 'n_steps': -1, 'name': name, 'mode': mode, 'atol': atol, 'rtol': rtol}
        if self.j_cb is not None:
            info['njev'] = jac.ncall
        results.append(info)
    return results