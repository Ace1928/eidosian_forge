from ..libmp.backend import xrange
from .calculus import defun
@defun
def adaptive_extrapolation(ctx, update, emfun, kwargs):
    option = kwargs.get
    if ctx._fixed_precision:
        tol = option('tol', ctx.eps * 2 ** 10)
    else:
        tol = option('tol', ctx.eps / 2 ** 10)
    verbose = option('verbose', False)
    maxterms = option('maxterms', ctx.dps * 10)
    method = set(option('method', 'r+s').split('+'))
    skip = option('skip', 0)
    steps = iter(option('steps', xrange(10, 10 ** 9, 10)))
    strict = option('strict')
    summer = []
    if 'd' in method or 'direct' in method:
        TRY_RICHARDSON = TRY_SHANKS = TRY_EULER_MACLAURIN = False
    else:
        TRY_RICHARDSON = 'r' in method or 'richardson' in method
        TRY_SHANKS = 's' in method or 'shanks' in method
        TRY_EULER_MACLAURIN = 'e' in method or 'euler-maclaurin' in method

        def init_levin(m):
            variant = kwargs.get('levin_variant', 'u')
            if isinstance(variant, str):
                if variant == 'all':
                    variant = ['u', 'v', 't']
                else:
                    variant = [variant]
            for s in variant:
                L = levin_class(method=m, variant=s)
                L.ctx = ctx
                L.name = m + '(' + s + ')'
                summer.append(L)
        if 'l' in method or 'levin' in method:
            init_levin('levin')
        if 'sidi' in method:
            init_levin('sidi')
        if 'a' in method or 'alternating' in method:
            L = cohen_alt_class()
            L.ctx = ctx
            L.name = 'alternating'
            summer.append(L)
    last_richardson_value = 0
    shanks_table = []
    index = 0
    step = 10
    partial = []
    best = ctx.zero
    orig = ctx.prec
    try:
        if 'workprec' in kwargs:
            ctx.prec = kwargs['workprec']
        elif TRY_RICHARDSON or TRY_SHANKS or len(summer) != 0:
            ctx.prec = (ctx.prec + 10) * 4
        else:
            ctx.prec += 30
        while 1:
            if index >= maxterms:
                break
            try:
                step = next(steps)
            except StopIteration:
                pass
            if verbose:
                print('-' * 70)
                print('Adding terms #%i-#%i' % (index, index + step))
            update(partial, xrange(index, index + step))
            index += step
            best = partial[-1]
            error = abs(best - partial[-2])
            if verbose:
                print('Direct error: %s' % ctx.nstr(error))
            if error <= tol:
                return best
            if TRY_RICHARDSON:
                value, maxc = ctx.richardson(partial)
                richardson_error = abs(value - last_richardson_value)
                if verbose:
                    print('Richardson error: %s' % ctx.nstr(richardson_error))
                if richardson_error <= tol:
                    return value
                last_richardson_value = value
                if ctx.eps * maxc > tol:
                    if verbose:
                        print('Ran out of precision for Richardson')
                    TRY_RICHARDSON = False
                if richardson_error < error:
                    error = richardson_error
                    best = value
            if TRY_SHANKS:
                shanks_table = ctx.shanks(partial, shanks_table, randomized=True)
                row = shanks_table[-1]
                if len(row) == 2:
                    est1 = row[-1]
                    shanks_error = 0
                else:
                    est1, maxc, est2 = (row[-1], abs(row[-2]), row[-3])
                    shanks_error = abs(est1 - est2)
                if verbose:
                    print('Shanks error: %s' % ctx.nstr(shanks_error))
                if shanks_error <= tol:
                    return est1
                if ctx.eps * maxc > tol:
                    if verbose:
                        print('Ran out of precision for Shanks')
                    TRY_SHANKS = False
                if shanks_error < error:
                    error = shanks_error
                    best = est1
            for L in summer:
                est, lerror = L.update_psum(partial)
                if verbose:
                    print('%s error: %s' % (L.name, ctx.nstr(lerror)))
                if lerror <= tol:
                    return est
                if lerror < error:
                    error = lerror
                    best = est
            if TRY_EULER_MACLAURIN:
                if ctx.almosteq(ctx.mpc(ctx.sign(partial[-1]) / ctx.sign(partial[-2])), -1):
                    if verbose:
                        print('NOT using Euler-Maclaurin: the series appears to be alternating, so numerical\n quadrature will most likely fail')
                    TRY_EULER_MACLAURIN = False
                else:
                    value, em_error = emfun(index, tol)
                    value += partial[-1]
                    if verbose:
                        print('Euler-Maclaurin error: %s' % ctx.nstr(em_error))
                    if em_error <= tol:
                        return value
                    if em_error < error:
                        best = value
    finally:
        ctx.prec = orig
    if strict:
        raise ctx.NoConvergence
    if verbose:
        print('Warning: failed to converge to target accuracy')
    return best