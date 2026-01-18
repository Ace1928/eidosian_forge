import functools
import re
from .ctx_base import StandardBaseContext
from .libmp.backend import basestring, BACKEND
from . import libmp
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import function_docs
from . import rational
from .ctx_mp_python import _mpf, _mpc, mpnumeric
def _nthroot(ctx, x, n):
    if hasattr(x, '_mpf_'):
        try:
            return ctx.make_mpf(libmp.mpf_nthroot(x._mpf_, n, *ctx._prec_rounding))
        except ComplexResult:
            if ctx.trap_complex:
                raise
            x = (x._mpf_, libmp.fzero)
    else:
        x = x._mpc_
    return ctx.make_mpc(libmp.mpc_nthroot(x, n, *ctx._prec_rounding))