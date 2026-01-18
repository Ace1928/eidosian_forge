import functools
import re
from .ctx_base import StandardBaseContext
from .libmp.backend import basestring, BACKEND
from . import libmp
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import function_docs
from . import rational
from .ctx_mp_python import _mpf, _mpc, mpnumeric
def _parse_prec(ctx, kwargs):
    if kwargs:
        if kwargs.get('exact'):
            return (0, 'f')
        prec, rounding = ctx._prec_rounding
        if 'rounding' in kwargs:
            rounding = kwargs['rounding']
        if 'prec' in kwargs:
            prec = kwargs['prec']
            if prec == ctx.inf:
                return (0, 'f')
            else:
                prec = int(prec)
        elif 'dps' in kwargs:
            dps = kwargs['dps']
            if dps == ctx.inf:
                return (0, 'f')
            prec = dps_to_prec(dps)
        return (prec, rounding)
    return ctx._prec_rounding