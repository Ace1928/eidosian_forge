import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
@classmethod
def _process_rvs_method(cls, method, r, c, n):
    known_methods = {None: cls._rvs_select(r, c, n), 'boyett': cls._rvs_boyett, 'patefield': cls._rvs_patefield}
    try:
        return known_methods[method]
    except KeyError:
        raise ValueError(f"'{method}' not recognized, must be one of {set(known_methods)}")