from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
class PythonMPContext(object):

    def __init__(ctx):
        ctx._prec_rounding = [53, round_nearest]
        ctx.mpf = type('mpf', (_mpf,), {})
        ctx.mpc = type('mpc', (_mpc,), {})
        ctx.mpf._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.mpc._ctxdata = [ctx.mpc, new, ctx._prec_rounding]
        ctx.mpf.context = ctx
        ctx.mpc.context = ctx
        ctx.constant = type('constant', (_constant,), {})
        ctx.constant._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.constant.context = ctx

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpf_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpc_ = v
        return a

    def default(ctx):
        ctx._prec = ctx._prec_rounding[0] = 53
        ctx._dps = 15
        ctx.trap_complex = False

    def _set_prec(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))
    prec = property(lambda ctx: ctx._prec, _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def convert(ctx, x, strings=True):
        """
        Converts *x* to an ``mpf`` or ``mpc``. If *x* is of type ``mpf``,
        ``mpc``, ``int``, ``float``, ``complex``, the conversion
        will be performed losslessly.

        If *x* is a string, the result will be rounded to the present
        working precision. Strings representing fractions or complex
        numbers are permitted.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> mpmathify(3.5)
            mpf('3.5')
            >>> mpmathify('2.1')
            mpf('2.1000000000000001')
            >>> mpmathify('3/4')
            mpf('0.75')
            >>> mpmathify('2+3j')
            mpc(real='2.0', imag='3.0')

        """
        if type(x) in ctx.types:
            return x
        if isinstance(x, int_types):
            return ctx.make_mpf(from_int(x))
        if isinstance(x, float):
            return ctx.make_mpf(from_float(x))
        if isinstance(x, complex):
            return ctx.make_mpc((from_float(x.real), from_float(x.imag)))
        if type(x).__module__ == 'numpy':
            return ctx.npconvert(x)
        if isinstance(x, numbers.Rational):
            try:
                x = rational.mpq(int(x.numerator), int(x.denominator))
            except:
                pass
        prec, rounding = ctx._prec_rounding
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return ctx.make_mpf(from_rational(p, q, prec))
        if strings and isinstance(x, basestring):
            try:
                _mpf_ = from_str(x, prec, rounding)
                return ctx.make_mpf(_mpf_)
            except ValueError:
                pass
        if hasattr(x, '_mpf_'):
            return ctx.make_mpf(x._mpf_)
        if hasattr(x, '_mpc_'):
            return ctx.make_mpc(x._mpc_)
        if hasattr(x, '_mpmath_'):
            return ctx.convert(x._mpmath_(prec, rounding))
        if type(x).__module__ == 'decimal':
            try:
                return ctx.make_mpf(from_Decimal(x, prec, rounding))
            except:
                pass
        return ctx._convert_fallback(x, strings)

    def npconvert(ctx, x):
        """
        Converts *x* to an ``mpf`` or ``mpc``. *x* should be a numpy
        scalar.
        """
        import numpy as np
        if isinstance(x, np.integer):
            return ctx.make_mpf(from_int(int(x)))
        if isinstance(x, np.floating):
            return ctx.make_mpf(from_npfloat(x))
        if isinstance(x, np.complexfloating):
            return ctx.make_mpc((from_npfloat(x.real), from_npfloat(x.imag)))
        raise TypeError('cannot create mpf from ' + repr(x))

    def isnan(ctx, x):
        """
        Return *True* if *x* is a NaN (not-a-number), or for a complex
        number, whether either the real or complex part is NaN;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isnan(3.14)
            False
            >>> isnan(nan)
            True
            >>> isnan(mpc(3.14,2.72))
            False
            >>> isnan(mpc(3.14,nan))
            True

        """
        if hasattr(x, '_mpf_'):
            return x._mpf_ == fnan
        if hasattr(x, '_mpc_'):
            return fnan in x._mpc_
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnan(x)
        raise TypeError('isnan() needs a number as input')

    def isinf(ctx, x):
        """
        Return *True* if the absolute value of *x* is infinite;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isinf(inf)
            True
            >>> isinf(-inf)
            True
            >>> isinf(3)
            False
            >>> isinf(3+4j)
            False
            >>> isinf(mpc(3,inf))
            True
            >>> isinf(mpc(inf,3))
            True

        """
        if hasattr(x, '_mpf_'):
            return x._mpf_ in (finf, fninf)
        if hasattr(x, '_mpc_'):
            re, im = x._mpc_
            return re in (finf, fninf) or im in (finf, fninf)
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isinf(x)
        raise TypeError('isinf() needs a number as input')

    def isnormal(ctx, x):
        """
        Determine whether *x* is "normal" in the sense of floating-point
        representation; that is, return *False* if *x* is zero, an
        infinity or NaN; otherwise return *True*. By extension, a
        complex number *x* is considered "normal" if its magnitude is
        normal::

            >>> from mpmath import *
            >>> isnormal(3)
            True
            >>> isnormal(0)
            False
            >>> isnormal(inf); isnormal(-inf); isnormal(nan)
            False
            False
            False
            >>> isnormal(0+0j)
            False
            >>> isnormal(0+3j)
            True
            >>> isnormal(mpc(2,nan))
            False
        """
        if hasattr(x, '_mpf_'):
            return bool(x._mpf_[1])
        if hasattr(x, '_mpc_'):
            re, im = x._mpc_
            re_normal = bool(re[1])
            im_normal = bool(im[1])
            if re == fzero:
                return im_normal
            if im == fzero:
                return re_normal
            return re_normal and im_normal
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return bool(x)
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnormal(x)
        raise TypeError('isnormal() needs a number as input')

    def isint(ctx, x, gaussian=False):
        """
        Return *True* if *x* is integer-valued; otherwise return
        *False*::

            >>> from mpmath import *
            >>> isint(3)
            True
            >>> isint(mpf(3))
            True
            >>> isint(3.2)
            False
            >>> isint(inf)
            False

        Optionally, Gaussian integers can be checked for::

            >>> isint(3+0j)
            True
            >>> isint(3+2j)
            False
            >>> isint(3+2j, gaussian=True)
            True

        """
        if isinstance(x, int_types):
            return True
        if hasattr(x, '_mpf_'):
            sign, man, exp, bc = xval = x._mpf_
            return bool(man and exp >= 0 or xval == fzero)
        if hasattr(x, '_mpc_'):
            re, im = x._mpc_
            rsign, rman, rexp, rbc = re
            isign, iman, iexp, ibc = im
            re_isint = rman and rexp >= 0 or re == fzero
            if gaussian:
                im_isint = iman and iexp >= 0 or im == fzero
                return re_isint and im_isint
            return re_isint and im == fzero
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return p % q == 0
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isint(x, gaussian)
        raise TypeError('isint() needs a number as input')

    def fsum(ctx, terms, absolute=False, squared=False):
        """
        Calculates a sum containing a finite number of terms (for infinite
        series, see :func:`~mpmath.nsum`). The terms will be converted to
        mpmath numbers. For len(terms) > 2, this function is generally
        faster and produces more accurate results than the builtin
        Python function :func:`sum`.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fsum([1, 2, 0.5, 7])
            mpf('10.5')

        With squared=True each term is squared, and with absolute=True
        the absolute value of each term is used.
        """
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        for term in terms:
            reval = imval = 0
            if hasattr(term, '_mpf_'):
                reval = term._mpf_
            elif hasattr(term, '_mpc_'):
                reval, imval = term._mpc_
            else:
                term = ctx.convert(term)
                if hasattr(term, '_mpf_'):
                    reval = term._mpf_
                elif hasattr(term, '_mpc_'):
                    reval, imval = term._mpc_
                else:
                    raise NotImplementedError
            if imval:
                if squared:
                    if absolute:
                        real.append(mpf_mul(reval, reval))
                        real.append(mpf_mul(imval, imval))
                    else:
                        reval, imval = mpc_pow_int((reval, imval), 2, prec + 10)
                        real.append(reval)
                        imag.append(imval)
                elif absolute:
                    real.append(mpc_abs((reval, imval), prec))
                else:
                    real.append(reval)
                    imag.append(imval)
            else:
                if squared:
                    reval = mpf_mul(reval, reval)
                elif absolute:
                    reval = mpf_abs(reval)
                real.append(reval)
        s = mpf_sum(real, prec, rnd, absolute)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def fdot(ctx, A, B=None, conjugate=False):
        """
        Computes the dot product of the iterables `A` and `B`,

        .. math ::

            \\sum_{k=0} A_k B_k.

        Alternatively, :func:`~mpmath.fdot` accepts a single iterable of pairs.
        In other words, ``fdot(A,B)`` and ``fdot(zip(A,B))`` are equivalent.
        The elements are automatically converted to mpmath numbers.

        With ``conjugate=True``, the elements in the second vector
        will be conjugated:

        .. math ::

            \\sum_{k=0} A_k \\overline{B_k}

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = [2, 1.5, 3]
            >>> B = [1, -1, 2]
            >>> fdot(A, B)
            mpf('6.5')
            >>> list(zip(A, B))
            [(2, 1), (1.5, -1), (3, 2)]
            >>> fdot(_)
            mpf('6.5')
            >>> A = [2, 1.5, 3j]
            >>> B = [1+j, 3, -1-j]
            >>> fdot(A, B)
            mpc(real='9.5', imag='-1.0')
            >>> fdot(A, B, conjugate=True)
            mpc(real='3.5', imag='-5.0')

        """
        if B is not None:
            A = zip(A, B)
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        hasattr_ = hasattr
        types = (ctx.mpf, ctx.mpc)
        for a, b in A:
            if type(a) not in types:
                a = ctx.convert(a)
            if type(b) not in types:
                b = ctx.convert(b)
            a_real = hasattr_(a, '_mpf_')
            b_real = hasattr_(b, '_mpf_')
            if a_real and b_real:
                real.append(mpf_mul(a._mpf_, b._mpf_))
                continue
            a_complex = hasattr_(a, '_mpc_')
            b_complex = hasattr_(b, '_mpc_')
            if a_real and b_complex:
                aval = a._mpf_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(aval, bre))
                imag.append(mpf_mul(aval, bim))
            elif b_real and a_complex:
                are, aim = a._mpc_
                bval = b._mpf_
                real.append(mpf_mul(are, bval))
                imag.append(mpf_mul(aim, bval))
            elif a_complex and b_complex:
                are, aim = a._mpc_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(are, bre))
                real.append(mpf_neg(mpf_mul(aim, bim)))
                imag.append(mpf_mul(are, bim))
                imag.append(mpf_mul(aim, bre))
            else:
                raise NotImplementedError
        s = mpf_sum(real, prec, rnd)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def _wrap_libmp_function(ctx, mpf_f, mpc_f=None, mpi_f=None, doc='<no doc>'):
        """
        Given a low-level mpf_ function, and optionally similar functions
        for mpc_ and mpi_, defines the function as a context method.

        It is assumed that the return type is the same as that of
        the input; the exception is that propagation from mpf to mpc is possible
        by raising ComplexResult.

        """

        def f(x, **kwargs):
            if type(x) not in ctx.types:
                x = ctx.convert(x)
            prec, rounding = ctx._prec_rounding
            if kwargs:
                prec = kwargs.get('prec', prec)
                if 'dps' in kwargs:
                    prec = dps_to_prec(kwargs['dps'])
                rounding = kwargs.get('rounding', rounding)
            if hasattr(x, '_mpf_'):
                try:
                    return ctx.make_mpf(mpf_f(x._mpf_, prec, rounding))
                except ComplexResult:
                    if ctx.trap_complex:
                        raise
                    return ctx.make_mpc(mpc_f((x._mpf_, fzero), prec, rounding))
            elif hasattr(x, '_mpc_'):
                return ctx.make_mpc(mpc_f(x._mpc_, prec, rounding))
            raise NotImplementedError('%s of a %s' % (name, type(x)))
        name = mpf_f.__name__[4:]
        f.__doc__ = function_docs.__dict__.get(name, 'Computes the %s of x' % doc)
        return f

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
        f_wrapped.__doc__ = function_docs.__dict__.get(name, f.__doc__)
        setattr(cls, name, f_wrapped)

    def _convert_param(ctx, x):
        if hasattr(x, '_mpc_'):
            v, im = x._mpc_
            if im != fzero:
                return (x, 'C')
        elif hasattr(x, '_mpf_'):
            v = x._mpf_
        else:
            if type(x) in int_types:
                return (int(x), 'Z')
            p = None
            if isinstance(x, tuple):
                p, q = x
            elif hasattr(x, '_mpq_'):
                p, q = x._mpq_
            elif isinstance(x, basestring) and '/' in x:
                p, q = x.split('/')
                p = int(p)
                q = int(q)
            if p is not None:
                if not p % q:
                    return (p // q, 'Z')
                return (ctx.mpq(p, q), 'Q')
            x = ctx.convert(x)
            if hasattr(x, '_mpc_'):
                v, im = x._mpc_
                if im != fzero:
                    return (x, 'C')
            elif hasattr(x, '_mpf_'):
                v = x._mpf_
            else:
                return (x, 'U')
        sign, man, exp, bc = v
        if man:
            if exp >= -4:
                if sign:
                    man = -man
                if exp >= 0:
                    return (int(man) << exp, 'Z')
                if exp >= -4:
                    p, q = (int(man), 1 << -exp)
                    return (ctx.mpq(p, q), 'Q')
            x = ctx.make_mpf(v)
            return (x, 'R')
        elif not exp:
            return (0, 'Z')
        else:
            return (x, 'U')

    def _mpf_mag(ctx, x):
        sign, man, exp, bc = x
        if man:
            return exp + bc
        if x == fzero:
            return ctx.ninf
        if x == finf or x == fninf:
            return ctx.inf
        return ctx.nan

    def mag(ctx, x):
        """
        Quick logarithmic magnitude estimate of a number. Returns an
        integer or infinity `m` such that `|x| <= 2^m`. It is not
        guaranteed that `m` is an optimal bound, but it will never
        be too large by more than 2 (and probably not more than 1).

        **Examples**

            >>> from mpmath import *
            >>> mp.pretty = True
            >>> mag(10), mag(10.0), mag(mpf(10)), int(ceil(log(10,2)))
            (4, 4, 4, 4)
            >>> mag(10j), mag(10+10j)
            (4, 5)
            >>> mag(0.01), int(ceil(log(0.01,2)))
            (-6, -6)
            >>> mag(0), mag(inf), mag(-inf), mag(nan)
            (-inf, +inf, +inf, nan)

        """
        if hasattr(x, '_mpf_'):
            return ctx._mpf_mag(x._mpf_)
        elif hasattr(x, '_mpc_'):
            r, i = x._mpc_
            if r == fzero:
                return ctx._mpf_mag(i)
            if i == fzero:
                return ctx._mpf_mag(r)
            return 1 + max(ctx._mpf_mag(r), ctx._mpf_mag(i))
        elif isinstance(x, int_types):
            if x:
                return bitcount(abs(x))
            return ctx.ninf
        elif isinstance(x, rational.mpq):
            p, q = x._mpq_
            if p:
                return 1 + bitcount(abs(p)) - bitcount(q)
            return ctx.ninf
        else:
            x = ctx.convert(x)
            if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
                return ctx.mag(x)
            else:
                raise TypeError('requires an mpf/mpc')