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
def _call_super_mom(fun):

    @wraps(fun)
    def wrapper(self, data, *args, **kwds):
        method = kwds.get('method', 'mle').lower()
        censored = isinstance(data, CensoredData)
        if method == 'mm' or (censored and data.num_censored() > 0):
            return super(type(self), self).fit(data, *args, **kwds)
        else:
            if censored:
                data = data._uncensored
            return fun(self, data, *args, **kwds)
    return wrapper