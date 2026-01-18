import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.

    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.
    """
    p = np.array(x0, copy=True)
    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)
    if fprime is not None:
        for iteration in range(maxiter):
            fval = np.asarray(func(p, *args))
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = np.asarray(fprime(p, *args))
            nz_der = fder != 0
            if not nz_der.any():
                break
            dp = fval[nz_der] / fder[nz_der]
            if fprime2 is not None:
                fder2 = np.asarray(fprime2(p, *args))
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
            p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol
            if not failures[nz_der].any():
                break
    else:
        dx = np.finfo(float).eps ** 0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = np.asarray(func(p, *args))
        q1 = np.asarray(func(p1, *args))
        active = np.ones_like(p, dtype=bool)
        for iteration in range(maxiter):
            nz_der = q1 != q0
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der
            failures[nz_der] = np.abs(dp) >= tol
            if not failures[nz_der].any():
                break
            p1, p = (p, p1)
            q0 = q1
            q1 = np.asarray(func(p1, *args))
    zero_der = ~nz_der & failures
    if zero_der.any():
        if fprime is None:
            nonzero_dp = p1 != p
            zero_der_nz_dp = zero_der & nonzero_dp
            if zero_der_nz_dp.any():
                rms = np.sqrt(sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2))
                warnings.warn(f'RMS of {rms:g} reached', RuntimeWarning, stacklevel=3)
        else:
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = f'{all_or_some:s} derivatives were zero'
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    elif failures.any():
        all_or_some = 'all' if failures.all() else 'some'
        msg = f'{all_or_some:s} failed to converge after {maxiter:d} iterations'
        if failures.all():
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    if full_output:
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        p = result(p, ~failures, zero_der)
    return p