import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
class MPIntervalContext(StandardBaseContext):

    def __init__(ctx):
        ctx.mpf = type('ivmpf', (ivmpf,), {})
        ctx.mpc = type('ivmpc', (ivmpc,), {})
        ctx._types = (ctx.mpf, ctx.mpc)
        ctx._constant = type('ivmpf_constant', (ivmpf_constant,), {})
        ctx._prec = [53]
        ctx._set_prec(53)
        ctx._constant._ctxdata = ctx.mpf._ctxdata = ctx.mpc._ctxdata = [ctx.mpf, new, ctx._prec]
        ctx._constant.ctx = ctx.mpf.ctx = ctx.mpc.ctx = ctx
        ctx.pretty = False
        StandardBaseContext.__init__(ctx)
        ctx._init_builtins()

    def _mpi(ctx, a, b=None):
        if b is None:
            return ctx.mpf(a)
        return ctx.mpf((a, b))

    def _init_builtins(ctx):
        ctx.one = ctx.mpf(1)
        ctx.zero = ctx.mpf(0)
        ctx.inf = ctx.mpf('inf')
        ctx.ninf = -ctx.inf
        ctx.nan = ctx.mpf('nan')
        ctx.j = ctx.mpc(0, 1)
        ctx.exp = ctx._wrap_mpi_function(libmp.mpi_exp, libmp.mpci_exp)
        ctx.sqrt = ctx._wrap_mpi_function(libmp.mpi_sqrt)
        ctx.ln = ctx._wrap_mpi_function(libmp.mpi_log, libmp.mpci_log)
        ctx.cos = ctx._wrap_mpi_function(libmp.mpi_cos, libmp.mpci_cos)
        ctx.sin = ctx._wrap_mpi_function(libmp.mpi_sin, libmp.mpci_sin)
        ctx.tan = ctx._wrap_mpi_function(libmp.mpi_tan)
        ctx.gamma = ctx._wrap_mpi_function(libmp.mpi_gamma, libmp.mpci_gamma)
        ctx.loggamma = ctx._wrap_mpi_function(libmp.mpi_loggamma, libmp.mpci_loggamma)
        ctx.rgamma = ctx._wrap_mpi_function(libmp.mpi_rgamma, libmp.mpci_rgamma)
        ctx.factorial = ctx._wrap_mpi_function(libmp.mpi_factorial, libmp.mpci_factorial)
        ctx.fac = ctx.factorial
        ctx.eps = ctx._constant(lambda prec, rnd: (0, MPZ_ONE, 1 - prec, 1))
        ctx.pi = ctx._constant(libmp.mpf_pi)
        ctx.e = ctx._constant(libmp.mpf_e)
        ctx.ln2 = ctx._constant(libmp.mpf_ln2)
        ctx.ln10 = ctx._constant(libmp.mpf_ln10)
        ctx.phi = ctx._constant(libmp.mpf_phi)
        ctx.euler = ctx._constant(libmp.mpf_euler)
        ctx.catalan = ctx._constant(libmp.mpf_catalan)
        ctx.glaisher = ctx._constant(libmp.mpf_glaisher)
        ctx.khinchin = ctx._constant(libmp.mpf_khinchin)
        ctx.twinprime = ctx._constant(libmp.mpf_twinprime)

    def _wrap_mpi_function(ctx, f_real, f_complex=None):

        def g(x, **kwargs):
            if kwargs:
                prec = kwargs.get('prec', ctx._prec[0])
            else:
                prec = ctx._prec[0]
            x = ctx.convert(x)
            if hasattr(x, '_mpi_'):
                return ctx.make_mpf(f_real(x._mpi_, prec))
            if hasattr(x, '_mpci_'):
                return ctx.make_mpc(f_complex(x._mpci_, prec))
            raise ValueError
        return g

    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:

            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                prec = ctx.prec
                try:
                    ctx.prec += 10
                    retval = f(ctx, *args, **kwargs)
                finally:
                    ctx.prec = prec
                return +retval
        else:
            f_wrapped = f
        setattr(cls, name, f_wrapped)

    def _set_prec(ctx, n):
        ctx._prec[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))
    prec = property(lambda ctx: ctx._prec[0], _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpi_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpci_ = v
        return a

    def _mpq(ctx, pq):
        p, q = pq
        a = libmp.from_rational(p, q, ctx.prec, round_floor)
        b = libmp.from_rational(p, q, ctx.prec, round_ceiling)
        return ctx.make_mpf((a, b))

    def convert(ctx, x):
        if isinstance(x, (ctx.mpf, ctx.mpc)):
            return x
        if isinstance(x, ctx._constant):
            return +x
        if isinstance(x, complex) or hasattr(x, '_mpc_'):
            re = ctx.convert(x.real)
            im = ctx.convert(x.imag)
            return ctx.mpc(re, im)
        if isinstance(x, basestring):
            v = mpi_from_str(x, ctx.prec)
            return ctx.make_mpf(v)
        if hasattr(x, '_mpi_'):
            a, b = x._mpi_
        else:
            try:
                a, b = x
            except (TypeError, ValueError):
                a = b = x
            if hasattr(a, '_mpi_'):
                a = a._mpi_[0]
            else:
                a = convert_mpf_(a, ctx.prec, round_floor)
            if hasattr(b, '_mpi_'):
                b = b._mpi_[1]
            else:
                b = convert_mpf_(b, ctx.prec, round_ceiling)
        if a == fnan or b == fnan:
            a = fninf
            b = finf
        assert mpf_le(a, b), 'endpoints must be properly ordered'
        return ctx.make_mpf((a, b))

    def nstr(ctx, x, n=5, **kwargs):
        x = ctx.convert(x)
        if hasattr(x, '_mpi_'):
            return libmp.mpi_to_str(x._mpi_, n, **kwargs)
        if hasattr(x, '_mpci_'):
            re = libmp.mpi_to_str(x._mpci_[0], n, **kwargs)
            im = libmp.mpi_to_str(x._mpci_[1], n, **kwargs)
            return '(%s + %s*j)' % (re, im)

    def mag(ctx, x):
        x = ctx.convert(x)
        if isinstance(x, ctx.mpc):
            return max(ctx.mag(x.real), ctx.mag(x.imag)) + 1
        a, b = libmp.mpi_abs(x._mpi_)
        sign, man, exp, bc = b
        if man:
            return exp + bc
        if b == fzero:
            return ctx.ninf
        if b == fnan:
            return ctx.nan
        return ctx.inf

    def isnan(ctx, x):
        return False

    def isinf(ctx, x):
        return x == ctx.inf

    def isint(ctx, x):
        x = ctx.convert(x)
        a, b = x._mpi_
        if a == b:
            sign, man, exp, bc = a
            if man:
                return exp >= 0
            return a == fzero
        return None

    def ldexp(ctx, x, n):
        a, b = ctx.convert(x)._mpi_
        a = libmp.mpf_shift(a, n)
        b = libmp.mpf_shift(b, n)
        return ctx.make_mpf((a, b))

    def absmin(ctx, x):
        return abs(ctx.convert(x)).a

    def absmax(ctx, x):
        return abs(ctx.convert(x)).b

    def atan2(ctx, y, x):
        y = ctx.convert(y)._mpi_
        x = ctx.convert(x)._mpi_
        return ctx.make_mpf(libmp.mpi_atan2(y, x, ctx.prec))

    def _convert_param(ctx, x):
        if isinstance(x, libmp.int_types):
            return (x, 'Z')
        if isinstance(x, tuple):
            p, q = x
            return (ctx.mpf(p) / ctx.mpf(q), 'R')
        x = ctx.convert(x)
        if isinstance(x, ctx.mpf):
            return (x, 'R')
        if isinstance(x, ctx.mpc):
            return (x, 'C')
        raise ValueError

    def _is_real_type(ctx, z):
        return isinstance(z, ctx.mpf) or isinstance(z, int_types)

    def _is_complex_type(ctx, z):
        return isinstance(z, ctx.mpc)

    def hypsum(ctx, p, q, types, coeffs, z, maxterms=6000, **kwargs):
        coeffs = list(coeffs)
        num = range(p)
        den = range(p, p + q)
        s = t = ctx.one
        k = 0
        while 1:
            for i in num:
                t *= coeffs[i] + k
            for i in den:
                t /= coeffs[i] + k
            k += 1
            t /= k
            t *= z
            s += t
            if t == 0:
                return s
            if k > maxterms:
                raise ctx.NoConvergence