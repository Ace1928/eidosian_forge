from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
from mpmath.rational import mpq
def _convert_tol(ctx, tol):
    if isinstance(tol, int_types):
        return from_int(tol)
    if isinstance(tol, float):
        return from_float(tol)
    if hasattr(tol, '_mpf_'):
        return tol._mpf_
    prec, rounding = ctx._prec_rounding
    if isinstance(tol, str):
        return from_str(tol, prec, rounding)
    raise ValueError('expected a real number, got %s' % tol)