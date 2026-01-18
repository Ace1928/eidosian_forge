import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class ode:
    """
    A generic interface class to numeric integrators.

    Solve an equation system :math:`y'(t) = f(t,y)` with (optional) ``jac = df/dy``.

    *Note*: The first two arguments of ``f(t, y, ...)`` are in the
    opposite order of the arguments in the system definition function used
    by `scipy.integrate.odeint`.

    Parameters
    ----------
    f : callable ``f(t, y, *f_args)``
        Right-hand side of the differential equation. t is a scalar,
        ``y.shape == (n,)``.
        ``f_args`` is set by calling ``set_f_params(*args)``.
        `f` should return a scalar, array or list (not a tuple).
    jac : callable ``jac(t, y, *jac_args)``, optional
        Jacobian of the right-hand side, ``jac[i,j] = d f[i] / d y[j]``.
        ``jac_args`` is set by calling ``set_jac_params(*args)``.

    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current variable values.

    See also
    --------
    odeint : an integrator with a simpler interface based on lsoda from ODEPACK
    quad : for finding the area under a curve

    Notes
    -----
    Available integrators are listed below. They can be selected using
    the `set_integrator` method.

    "vode"

        Real-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        implicit Adams method (for non-stiff problems) and a method based on
        backward differentiation formulas (BDF) (for stiff problems).

        Source: http://www.netlib.org/ode/vode.f

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "vode" integrator at the same time.

        This integrator accepts the following parameters in `set_integrator`
        method of the `ode` class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - lband : None or int
        - uband : None or int
          Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
          Setting these requires your jac routine to return the jacobian
          in packed format, jac_packed[i-j+uband, j] = jac[i,j]. The
          dimension of the matrix must be (lband+uband+1, len(y)).
        - method: 'adams' or 'bdf'
          Which solver to use, Adams (non-stiff) or BDF (stiff)
        - with_jacobian : bool
          This option is only considered when the user has not supplied a
          Jacobian function and has not indicated (by setting either band)
          that the Jacobian is banded. In this case, `with_jacobian` specifies
          whether the iteration method of the ODE solver's correction step is
          chord iteration with an internally generated full Jacobian or
          functional iteration with no Jacobian.
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
        - min_step : float
        - max_step : float
          Limits for the step sizes used by the integrator.
        - order : int
          Maximum order used by the integrator,
          order <= 12 for Adams, <= 5 for BDF.

    "zvode"

        Complex-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        implicit Adams method (for non-stiff problems) and a method based on
        backward differentiation formulas (BDF) (for stiff problems).

        Source: http://www.netlib.org/ode/zvode.f

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "zvode" integrator at the same time.

        This integrator accepts the same parameters in `set_integrator`
        as the "vode" solver.

        .. note::

            When using ZVODE for a stiff system, it should only be used for
            the case in which the function f is analytic, that is, when each f(i)
            is an analytic function of each y(j). Analyticity means that the
            partial derivative df(i)/dy(j) is a unique complex number, and this
            fact is critical in the way ZVODE solves the dense or banded linear
            systems that arise in the stiff case. For a complex stiff ODE system
            in which f is not analytic, ZVODE is likely to have convergence
            failures, and for this problem one should instead use DVODE on the
            equivalent real system (in the real and imaginary parts of y).

    "lsoda"

        Real-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        automatic method switching between implicit Adams method (for non-stiff
        problems) and a method based on backward differentiation formulas (BDF)
        (for stiff problems).

        Source: http://www.netlib.org/odepack

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "lsoda" integrator at the same time.

        This integrator accepts the following parameters in `set_integrator`
        method of the `ode` class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - lband : None or int
        - uband : None or int
          Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
          Setting these requires your jac routine to return the jacobian
          in packed format, jac_packed[i-j+uband, j] = jac[i,j].
        - with_jacobian : bool
          *Not used.*
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
        - min_step : float
        - max_step : float
          Limits for the step sizes used by the integrator.
        - max_order_ns : int
          Maximum order used in the nonstiff case (default 12).
        - max_order_s : int
          Maximum order used in the stiff case (default 5).
        - max_hnil : int
          Maximum number of messages reporting too small step size (t + h = t)
          (default 0)
        - ixpr : int
          Whether to generate extra printing at method switches (default False).

    "dopri5"

        This is an explicit runge-kutta method of order (4)5 due to Dormand &
        Prince (with stepsize control and dense output).

        Authors:

            E. Hairer and G. Wanner
            Universite de Geneve, Dept. de Mathematiques
            CH-1211 Geneve 24, Switzerland
            e-mail:  ernst.hairer@math.unige.ch, gerhard.wanner@math.unige.ch

        This code is described in [HNW93]_.

        This integrator accepts the following parameters in set_integrator()
        method of the ode class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
        - max_step : float
        - safety : float
          Safety factor on new step selection (default 0.9)
        - ifactor : float
        - dfactor : float
          Maximum factor to increase/decrease step size by in one step
        - beta : float
          Beta parameter for stabilised step size control.
        - verbosity : int
          Switch for printing messages (< 0 for no messages).

    "dop853"

        This is an explicit runge-kutta method of order 8(5,3) due to Dormand
        & Prince (with stepsize control and dense output).

        Options and references the same as "dopri5".

    Examples
    --------

    A problem to integrate and the corresponding jacobian:

    >>> from scipy.integrate import ode
    >>>
    >>> y0, t0 = [1.0j, 2.0], 0
    >>>
    >>> def f(t, y, arg1):
    ...     return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
    >>> def jac(t, y, arg1):
    ...     return [[1j*arg1, 1], [0, -arg1*2*y[1]]]

    The integration:

    >>> r = ode(f, jac).set_integrator('zvode', method='bdf')
    >>> r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
    >>> t1 = 10
    >>> dt = 1
    >>> while r.successful() and r.t < t1:
    ...     print(r.t+dt, r.integrate(r.t+dt))
    1 [-0.71038232+0.23749653j  0.40000271+0.j        ]
    2.0 [0.19098503-0.52359246j 0.22222356+0.j        ]
    3.0 [0.47153208+0.52701229j 0.15384681+0.j        ]
    4.0 [-0.61905937+0.30726255j  0.11764744+0.j        ]
    5.0 [0.02340997-0.61418799j 0.09523835+0.j        ]
    6.0 [0.58643071+0.339819j 0.08000018+0.j      ]
    7.0 [-0.52070105+0.44525141j  0.06896565+0.j        ]
    8.0 [-0.15986733-0.61234476j  0.06060616+0.j        ]
    9.0 [0.64850462+0.15048982j 0.05405414+0.j        ]
    10.0 [-0.38404699+0.56382299j  0.04878055+0.j        ]

    References
    ----------
    .. [HNW93] E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary
        Differential Equations i. Nonstiff Problems. 2nd edition.
        Springer Series in Computational Mathematics,
        Springer-Verlag (1993)

    """

    def __init__(self, f, jac=None):
        self.stiff = 0
        self.f = f
        self.jac = jac
        self.f_params = ()
        self.jac_params = ()
        self._y = []

    @property
    def y(self):
        return self._y

    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        if isscalar(y):
            y = [y]
        n_prev = len(self._y)
        if not n_prev:
            self.set_integrator('')
        self._y = asarray(y, self._integrator.scalar)
        self.t = t
        self._integrator.reset(len(self._y), self.jac is not None)
        return self

    def set_integrator(self, name, **integrator_params):
        """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator.
        **integrator_params
            Additional parameters for the integrator.
        """
        integrator = find_integrator(name)
        if integrator is None:
            message = f'No integrator name match with {name!r} or is not available.'
            warnings.warn(message, stacklevel=2)
        else:
            self._integrator = integrator(**integrator_params)
            if not len(self._y):
                self.t = 0.0
                self._y = array([0.0], self._integrator.scalar)
            self._integrator.reset(len(self._y), self.jac is not None)
        return self

    def integrate(self, t, step=False, relax=False):
        """Find y=y(t), set y as an initial condition, and return y.

        Parameters
        ----------
        t : float
            The endpoint of the integration step.
        step : bool
            If True, and if the integrator supports the step method,
            then perform a single integration step and return.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.
        relax : bool
            If True and if the integrator supports the run_relax method,
            then integrate until t_1 >= t and return. ``relax`` is not
            referenced if ``step=True``.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.

        Returns
        -------
        y : float
            The integrated value at t
        """
        if step and self._integrator.supports_step:
            mth = self._integrator.step
        elif relax and self._integrator.supports_run_relax:
            mth = self._integrator.run_relax
        else:
            mth = self._integrator.run
        try:
            self._y, self.t = mth(self.f, self.jac or (lambda: None), self._y, self.t, t, self.f_params, self.jac_params)
        except SystemError as e:
            raise ValueError('Function to integrate must not return a tuple.') from e
        return self._y

    def successful(self):
        """Check if integration was successful."""
        try:
            self._integrator
        except AttributeError:
            self.set_integrator('')
        return self._integrator.success == 1

    def get_return_code(self):
        """Extracts the return code for the integration to enable better control
        if the integration fails.

        In general, a return code > 0 implies success, while a return code < 0
        implies failure.

        Notes
        -----
        This section describes possible return codes and their meaning, for available
        integrators that can be selected by `set_integrator` method.

        "vode"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call. (Perhaps wrong MF.)
        -2           Excess accuracy requested. (Tolerances too small.)
        -3           Illegal input detected. (See printed message.)
        -4           Repeated error test failures. (Check all input.)
        -5           Repeated convergence failures. (Perhaps bad Jacobian
                     supplied or wrong choice of MF or tolerances.)
        -6           Error weight became zero during problem. (Solution
                     component i vanished, and ATOL or ATOL(i) = 0.)
        ===========  =======

        "zvode"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call. (Perhaps wrong MF.)
        -2           Excess accuracy requested. (Tolerances too small.)
        -3           Illegal input detected. (See printed message.)
        -4           Repeated error test failures. (Check all input.)
        -5           Repeated convergence failures. (Perhaps bad Jacobian
                     supplied or wrong choice of MF or tolerances.)
        -6           Error weight became zero during problem. (Solution
                     component i vanished, and ATOL or ATOL(i) = 0.)
        ===========  =======

        "dopri5"

        ===========  =======
        Return Code  Message
        ===========  =======
        1            Integration successful.
        2            Integration successful (interrupted by solout).
        -1           Input is not consistent.
        -2           Larger nsteps is needed.
        -3           Step size becomes too small.
        -4           Problem is probably stiff (interrupted).
        ===========  =======

        "dop853"

        ===========  =======
        Return Code  Message
        ===========  =======
        1            Integration successful.
        2            Integration successful (interrupted by solout).
        -1           Input is not consistent.
        -2           Larger nsteps is needed.
        -3           Step size becomes too small.
        -4           Problem is probably stiff (interrupted).
        ===========  =======

        "lsoda"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call (perhaps wrong Dfun type).
        -2           Excess accuracy requested (tolerances too small).
        -3           Illegal input detected (internal error).
        -4           Repeated error test failures (internal error).
        -5           Repeated convergence failures (perhaps bad Jacobian or tolerances).
        -6           Error weight became zero during problem.
        -7           Internal workspace insufficient to finish (internal error).
        ===========  =======
        """
        try:
            self._integrator
        except AttributeError:
            self.set_integrator('')
        return self._integrator.istate

    def set_f_params(self, *args):
        """Set extra parameters for user-supplied function f."""
        self.f_params = args
        return self

    def set_jac_params(self, *args):
        """Set extra parameters for user-supplied function jac."""
        self.jac_params = args
        return self

    def set_solout(self, solout):
        """
        Set callable to be called at every successful integration step.

        Parameters
        ----------
        solout : callable
            ``solout(t, y)`` is called at each internal integrator step,
            t is a scalar providing the current independent position
            y is the current solution ``y.shape == (n,)``
            solout should return -1 to stop integration
            otherwise it should return None or 0

        """
        if self._integrator.supports_solout:
            self._integrator.set_solout(solout)
            if self._y is not None:
                self._integrator.reset(len(self._y), self.jac is not None)
        else:
            raise ValueError('selected integrator does not support solout, choose another one')