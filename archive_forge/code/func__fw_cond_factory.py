import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def _fw_cond_factory(self, ri, rtol=1e-14):
    rxn = self.rxns[ri]

    def fw_cond(x, p):
        precip_stoich_coeff, precip_idx = rxn.precipitate_stoich(self.substances)[1:3]
        q = rxn.Q(self.substances, self.dissolved(x))
        k = rxn.equilibrium_constant()
        if precip_stoich_coeff > 0:
            return q * (1 + rtol) < k
        elif precip_stoich_coeff < 0:
            return q > k * (1 + rtol)
        else:
            raise NotImplementedError
    return fw_cond