import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def _mpq(ctx, pq):
    p, q = pq
    a = libmp.from_rational(p, q, ctx.prec, round_floor)
    b = libmp.from_rational(p, q, ctx.prec, round_ceiling)
    return ctx.make_mpf((a, b))