import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
@cached_property
def _skewnorm_odd_moments(self):
    skewnorm_odd_moments = {1: Polynomial([1]), 3: Polynomial([3, -1]), 5: Polynomial([15, -10, 3]), 7: Polynomial([105, -105, 63, -15]), 9: Polynomial([945, -1260, 1134, -540, 105]), 11: Polynomial([10395, -17325, 20790, -14850, 5775, -945]), 13: Polynomial([135135, -270270, 405405, -386100, 225225, -73710, 10395]), 15: Polynomial([2027025, -4729725, 8513505, -10135125, 7882875, -3869775, 1091475, -135135]), 17: Polynomial([34459425, -91891800, 192972780, -275675400, 268017750, -175429800, 74220300, -18378360, 2027025]), 19: Polynomial([654729075, -1964187225, 4714049340, -7856748900, 9166207050, -7499623950, 4230557100, -1571349780, 346621275, -34459425])}
    return skewnorm_odd_moments