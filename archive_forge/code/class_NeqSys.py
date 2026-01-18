from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
class NeqSys(_NeqSysBase):
    """Represents a system of non-linear equations.

    This class provides a unified interface to:

    - scipy.optimize.root
    - NLEQ2
    - KINSOL
    - mpmath
    - levmar

    Parameters
    ----------
    nf : int
        Number of functions.
    nx : int
        Number of independent variables.
    f : callback
        Function to solve for. Signature ``f(x) -> y`` where ``len(x) == nx``
        and ``len(y) == nf``.
    jac : callback or None (default)
        Jacobian matrix (dfdy).
    band : tuple (default: None)
        Number of sub- and super-diagonals in jacobian.
    names : iterable of str (default: None)
        Names of variables, used for plotting and for referencing by name.
    param_names : iterable of strings (default: None)
        Names of the parameters, used for referencing parameters by name.
    x_by_name : bool, default: ``False``
        Will values for *x* be referred to by name (in dictionaries)
        instead of by index (in arrays)?
    par_by_name : bool, default: ``False``
        Will values for parameters be referred to by name (in dictionaries)
        instead of by index (in arrays)?
    latex_names : iterable of str, optional
        Names of variables in LaTeX format.
    latex_param_names : iterable of str, optional
        Names of parameters in LaTeX format.
    pre_processors : iterable of callables (optional)
        (Forward) transformation of user-input to :py:meth:`solve`
        signature: ``f(x1[:], params1[:]) -> x2[:], params2[:]``.
        Insert at beginning.
    post_processors : iterable of callables (optional)
        (Backward) transformation of result from :py:meth:`solve`
        signature: ``f(x2[:], params2[:]) -> x1[:], params1[:]``.
        Insert at end.
    internal_x0_cb : callback (optional)
        callback with signature ``f(x[:], p[:]) -> x0[:]``
        if not specified, ``x`` from ``self.pre_processors`` will be used.

    Examples
    --------
    >>> neqsys = NeqSys(2, 2, lambda x, p: [(x[0] - x[1])**p[0]/2 + x[0] - 1,
    ...                                     (x[1] - x[0])**p[0]/2 + x[1]])
    >>> x, sol = neqsys.solve([1, 0], [3])
    >>> assert sol['success']
    >>> print(x)  # doctest: +NORMALIZE_WHITESPACE
    [ 0.8411639  0.1588361]

    See Also
    --------
    pyneqsys.symbolic.SymbolicSys : use a CAS (SymPy by default) to derive
                                    the jacobian.
    """

    def __init__(self, nf, nx=None, f=None, jac=None, band=None, pre_processors=None, post_processors=None, internal_x0_cb=None, **kwargs):
        super(NeqSys, self).__init__(**kwargs)
        if nx is None:
            nx = len(self.names)
        if f is None:
            raise ValueError('A callback for f must be provided')
        if nf < nx:
            raise ValueError('Under-determined system')
        self.nf, self.nx = (nf, nx)
        self.f_cb = _ensure_3args(f)
        self.j_cb = _ensure_3args(jac)
        self.band = band
        self.pre_processors = pre_processors or []
        self.post_processors = post_processors or []
        self.internal_x0_cb = internal_x0_cb

    def pre_process(self, x0, params=()):
        """ Used internally for transformation of variables. """
        if self.x_by_name and isinstance(x0, dict):
            x0 = [x0[k] for k in self.names]
        if self.par_by_name and isinstance(params, dict):
            params = [params[k] for k in self.param_names]
        for pre_processor in self.pre_processors:
            x0, params = pre_processor(x0, params)
        return (x0, np.atleast_1d(params))

    def post_process(self, xout, params_out):
        """ Used internally for transformation of variables. """
        for post_processor in self.post_processors:
            xout, params_out = post_processor(xout, params_out)
        return (xout, params_out)

    def solve(self, x0, params=(), internal_x0=None, solver=None, attached_solver=None, **kwargs):
        """ Solve with user specified ``solver`` choice.

        Parameters
        ----------
        x0: 1D array of floats
            Guess (subject to ``self.post_processors``)
        params: 1D array_like of floats
            Parameters (subject to ``self.post_processors``)
        internal_x0: 1D array of floats
            When given it overrides (processed) ``x0``. ``internal_x0`` is not
            subject to ``self.post_processors``.
        solver: str or callable or None or iterable of such
            if str: uses _solve_``solver``(\\*args, \\*\\*kwargs).
            if ``None``: chooses from PYNEQSYS_SOLVER environment variable.
            if iterable: chain solving.
        attached_solver: callable factory
            Invokes: solver = attached_solver(self).

        Returns
        -------
        array:
            solution vector (post-processed by self.post_processors)
        dict:
            info dictionary containing 'success', 'nfev', 'njev' etc.

        Examples
        --------
        >>> neqsys = NeqSys(2, 2, lambda x, p: [
        ...     (x[0] - x[1])**p[0]/2 + x[0] - 1,
        ...     (x[1] - x[0])**p[0]/2 + x[1]
        ... ])
        >>> x, sol = neqsys.solve([1, 0], [3], solver=(None, 'mpmath'))
        >>> assert sol['success']
        >>> print(x)  # doctest: +NORMALIZE_WHITESPACE
        [0.841163901914009663684741869855]
        [0.158836098085990336315258130144]

        """
        if not isinstance(solver, (tuple, list)):
            solver = [solver]
        if not isinstance(attached_solver, (tuple, list)):
            attached_solver = [attached_solver] + [None] * (len(solver) - 1)
        _x0, self.internal_params = self.pre_process(x0, params)
        for solv, attached_solv in zip(solver, attached_solver):
            if internal_x0 is not None:
                _x0 = internal_x0
            elif self.internal_x0_cb is not None:
                _x0 = self.internal_x0_cb(x0, params)
            nfo = self._get_solver_cb(solv, attached_solv)(_x0, **kwargs)
            _x0 = nfo['x'].copy()
        self.internal_x = _x0
        x0 = self.post_process(self.internal_x, self.internal_params)[0]
        return (x0, nfo)

    def _solve_scipy(self, intern_x0, tol=1e-08, method=None, **kwargs):
        """ Uses ``scipy.optimize.root``

        See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

        Parameters
        ----------
        intern_x0: array_like
            initial guess
        tol: float
            Tolerance
        method: str
            What method to use. Defaults to ``'lm'`` if ``self.nf > self.nx`` otherwise ``'hybr'``.

        """
        from scipy.optimize import root
        if method is None:
            if self.nf > self.nx:
                method = 'lm'
            elif self.nf == self.nx:
                method = 'hybr'
            else:
                raise ValueError('Underdetermined problem')
        if 'band' in kwargs:
            raise ValueError("Set 'band' at initialization instead.")
        if 'args' in kwargs:
            raise ValueError("Set 'args' as params in initialization instead.")
        new_kwargs = kwargs.copy()
        if self.band is not None:
            warnings.warn('Band argument ignored (see SciPy docs)')
            new_kwargs['band'] = self.band
        new_kwargs['args'] = self.internal_params
        return root(self.f_cb, intern_x0, jac=self.j_cb, method=method, tol=tol, **new_kwargs)

    def _solve_nleq2(self, intern_x0, tol=1e-08, method=None, **kwargs):
        from pynleq2 import solve

        def f_cb(x, ierr):
            f_cb.nfev += 1
            return (self.f_cb(x, self.internal_params), ierr)
        f_cb.nfev = 0

        def j_cb(x, ierr):
            j_cb.njev += 1
            return (self.j_cb(x, self.internal_params), ierr)
        j_cb.njev = 0
        x, ierr = solve(f_cb, j_cb, intern_x0, **kwargs)
        return {'x': x, 'fun': np.asarray(f_cb(x, 0)), 'success': ierr == 0, 'nfev': f_cb.nfev, 'njev': j_cb.njev, 'ierr': ierr}

    def _solve_kinsol(self, intern_x0, **kwargs):
        import pykinsol

        def _f(x, fout):
            res = self.f_cb(x, self.internal_params)
            fout[:] = res

        def _j(x, Jout, fx):
            res = self.j_cb(x, self.internal_params)
            Jout[:, :] = res[:, :]
        return pykinsol.solve(_f, _j, intern_x0, **kwargs)

    def _solve_mpmath(self, intern_x0, dps=30, tol=None, maxsteps=None, **kwargs):
        import mpmath
        from mpmath.calculus.optimization import MDNewton
        mp = mpmath.mp
        mp.dps = dps

        def _mpf(val):
            try:
                return mp.mpf(val)
            except TypeError:
                return mp.mpf(float(val))
        intern_p = tuple((_mpf(_p) for _p in self.internal_params))
        maxsteps = maxsteps or MDNewton.maxsteps
        tol = tol or mp.eps * 1024

        def f_cb(*x):
            f_cb.nfev += 1
            return self.f_cb(x, intern_p)
        f_cb.nfev = 0
        if self.j_cb is not None:

            def j_cb(*x):
                j_cb.njev += 1
                return self.j_cb(x, intern_p)
            j_cb.njev = 0
            kwargs['J'] = j_cb
        intern_x0 = tuple((_mpf(_x) for _x in intern_x0))
        iters = MDNewton(mp, f_cb, intern_x0, norm=mp.norm, verbose=False, **kwargs)
        i = 0
        success = False
        for x, err in iters:
            i += 1
            lim = tol * max(mp.norm(x), 1)
            if err < lim:
                success = True
                break
            if i >= maxsteps:
                break
        result = {'x': x, 'success': success, 'nfev': f_cb.nfev, 'nit': i}
        if self.j_cb is not None:
            result['njev'] = j_cb.njev
        return result

    def _solve_ipopt(self, intern_x0, **kwargs):
        import warnings
        from ipopt import minimize_ipopt
        warnings.warn('ipopt interface has not yet undergone thorough testing.')

        def f_cb(x):
            f_cb.nfev += 1
            return np.sum(np.abs(self.f_cb(x, self.internal_params)))
        f_cb.nfev = 0
        if self.j_cb is not None:

            def j_cb(x):
                j_cb.njev += 1
                return self.j_cb(x, self.internal_params)
            j_cb.njev = 0
            kwargs['jac'] = j_cb
        return minimize_ipopt(f_cb, intern_x0, **kwargs)

    def _solve_levmar(self, intern_x0, tol=1e-08, **kwargs):
        import warnings
        import levmar
        if 'eps1' in kwargs or 'eps2' in kwargs or 'eps3' in kwargs:
            pass
        else:
            kwargs['eps1'] = kwargs['eps2'] = kwargs['eps3'] = tol

        def _f(*args):
            return np.asarray(self.f_cb(*args))

        def _j(*args):
            return np.asarray(self.j_cb(*args))
        _x0 = np.asarray(intern_x0)
        _y0 = np.zeros(self.nf)
        with warnings.catch_warnings(record=True) as wrns:
            warnings.simplefilter('always')
            p_opt, p_cov, info = levmar.levmar(_f, _x0, _y0, args=(self.internal_params,), jacf=_j, **kwargs)
        success = len(wrns) == 0 and np.all(np.abs(_f(p_opt, self.internal_params)) < tol)
        for w in wrns:
            raise w
        e2p0, (e2, infJTe, Dp2, mu_maxJTJii), nit, reason, nfev, njev, nlinsolv = info
        return {'x': p_opt, 'cov': p_cov, 'nfev': nfev, 'njev': njev, 'nit': nit, 'message': reason, 'nlinsolv': nlinsolv, 'success': success}