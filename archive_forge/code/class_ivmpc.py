import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
class ivmpc(object):

    def __new__(cls, re=0, im=0):
        re = cls.ctx.convert(re)
        im = cls.ctx.convert(im)
        y = new(cls)
        y._mpci_ = (re._mpi_, im._mpi_)
        return y

    def __hash__(self):
        (a, b), (c, d) = self._mpci_
        if a == b and c == d:
            return mpc_hash((a, c))
        else:
            return hash(self._mpci_)

    def __repr__(s):
        if s.ctx.pretty:
            return str(s)
        return 'iv.mpc(%s, %s)' % (repr(s.real), repr(s.imag))

    def __str__(s):
        return '(%s + %s*j)' % (str(s.real), str(s.imag))

    @property
    def a(self):
        (a, b), (c, d) = self._mpci_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        (a, b), (c, d) = self._mpci_
        return self.ctx.make_mpf((b, b))

    @property
    def c(self):
        (a, b), (c, d) = self._mpci_
        return self.ctx.make_mpf((c, c))

    @property
    def d(self):
        (a, b), (c, d) = self._mpci_
        return self.ctx.make_mpf((d, d))

    @property
    def real(s):
        return s.ctx.make_mpf(s._mpci_[0])

    @property
    def imag(s):
        return s.ctx.make_mpf(s._mpci_[1])

    def conjugate(s):
        a, b = s._mpci_
        return s.ctx.make_mpc((a, mpf_neg(b)))

    def overlap(s, t):
        t = s.ctx.convert(t)
        real_overlap = s.a <= t.a <= s.b or s.a <= t.b <= s.b or t.a <= s.a <= t.b or (t.a <= s.b <= t.b)
        imag_overlap = s.c <= t.c <= s.d or s.c <= t.d <= s.d or t.c <= s.c <= t.d or (t.c <= s.d <= t.d)
        return real_overlap and imag_overlap

    def __contains__(s, t):
        t = s.ctx.convert(t)
        return t.real in s.real and t.imag in s.imag

    def _compare(s, t, ne=False):
        if not isinstance(t, s.ctx._types):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        if hasattr(t, '_mpi_'):
            tval = (t._mpi_, mpi_zero)
        elif hasattr(t, '_mpci_'):
            tval = t._mpci_
        if ne:
            return s._mpci_ != tval
        return s._mpci_ == tval

    def __eq__(s, t):
        return s._compare(t)

    def __ne__(s, t):
        return s._compare(t, True)

    def __lt__(s, t):
        raise TypeError('complex intervals cannot be ordered')
    __le__ = __gt__ = __ge__ = __lt__

    def __neg__(s):
        return s.ctx.make_mpc(mpci_neg(s._mpci_, s.ctx.prec))

    def __pos__(s):
        return s.ctx.make_mpc(mpci_pos(s._mpci_, s.ctx.prec))

    def __abs__(s):
        return s.ctx.make_mpf(mpci_abs(s._mpci_, s.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)