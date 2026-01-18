import cupy
def _minimize_scalar_bounded(func, bounds, args=(), xatol=1e-05, maxiter=500, disp=0, **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    maxfun = maxiter
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds
    if x1 > x2:
        raise ValueError('The lower bound exceeds the upper bound.')
    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'
    sqrt_eps = cupy.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - cupy.sqrt(5.0))
    a, b = (x1, x2)
    fulc = a + golden_mean * (b - a)
    nfc, xf = (fulc, fulc)
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = cupy.inf
    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * cupy.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    if disp > 2:
        print(' ')
        print(header)
        print('%5.0f   %12.6g %12.6g %s' % (fmin_data + (step,)))
    while cupy.abs(xf - xm) > tol2 - 0.5 * (b - a):
        golden = 1
        if cupy.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = cupy.abs(q)
            r = e
            e = rat
            if cupy.abs(p) < cupy.abs(0.5 * q * r) and p > q * (a - xf) and (p < q * (b - xf)):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'
                if x - a < tol2 or b - x < tol2:
                    si = cupy.sign(xm - xf) + (xm - xf == 0)
                    rat = tol1 * si
            else:
                golden = 1
        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
            step = '       golden'
        si = cupy.sign(rat) + (rat == 0)
        x = xf + si * cupy.maximum(cupy.abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print('%5.0f   %12.6g %12.6g %s' % (fmin_data + (step,)))
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = (nfc, fnfc)
            nfc, fnfc = (xf, fx)
            xf, fx = (x, fu)
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fnfc or nfc == xf:
                fulc, ffulc = (nfc, fnfc)
                nfc, fnfc = (x, fu)
            elif fu <= ffulc or fulc == xf or fulc == nfc:
                fulc, ffulc = (x, fu)
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * cupy.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            flag = 1
            break
    if cupy.isnan(xf) or cupy.isnan(fx) or cupy.isnan(fu):
        flag = 2
    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)
    result = OptimizeResult(fun=fval, status=flag, success=flag == 0, message={0: 'Solution found.', 1: 'Maximum number of function calls reached.', 2: _status_message['nan']}.get(flag, ''), x=xf, nfev=num, nit=num)
    return result