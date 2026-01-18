import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def _SymbolicSys_from_NumSys(self, NS, conds, rref_equil, rref_preserv, new_eq_params=True):
    from pyneqsys.symbolic import SymbolicSys
    import sympy as sp
    ns = NS(self, backend=sp, rref_equil=rref_equil, rref_preserv=rref_preserv, precipitates=conds, new_eq_params=new_eq_params)
    symb_kw = {}
    if ns.pre_processor is not None:
        symb_kw['pre_processors'] = [ns.pre_processor]
    if ns.post_processor is not None:
        symb_kw['post_processors'] = [ns.post_processor]
    if ns.internal_x0_cb is not None:
        symb_kw['internal_x0_cb'] = ns.internal_x0_cb
    return SymbolicSys.from_callback(ns.f, self.ns, nparams=self.ns + (self.nr if new_eq_params else 0), **symb_kw)