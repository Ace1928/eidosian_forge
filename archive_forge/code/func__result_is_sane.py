import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def _result_is_sane(self, init_concs, x, rtol=1e-09):
    sc_upper_bounds = np.array(self.upper_conc_bounds(init_concs))
    neg_conc, too_much = (np.any(x < 0), np.any(x > sc_upper_bounds * (1 + rtol)))
    if neg_conc or too_much:
        if neg_conc:
            warnings.warn('Negative concentration')
        if too_much:
            warnings.warn('Too much of at least one component')
        return False
    return True