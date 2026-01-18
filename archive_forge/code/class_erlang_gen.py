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
class erlang_gen(gamma_gen):
    """An Erlang continuous random variable.

    %(before_notes)s

    See Also
    --------
    gamma

    Notes
    -----
    The Erlang distribution is a special case of the Gamma distribution, with
    the shape parameter `a` an integer.  Note that this restriction is not
    enforced by `erlang`. It will, however, generate a warning the first time
    a non-integer value is used for the shape parameter.

    Refer to `gamma` for examples.

    """

    def _argcheck(self, a):
        allint = np.all(np.floor(a) == a)
        if not allint:
            message = f'The shape parameter of the erlang distribution has been given a non-integer value {a!r}.'
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        return a > 0

    def _shape_info(self):
        return [_ShapeInfo('a', True, (1, np.inf), (True, False))]

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        a = int(4.0 / (1e-08 + _skew(data) ** 2))
        return super(gamma_gen, self)._fitstart(data, args=(a,))

    @extend_notes_in_docstring(rv_continuous, notes='        The Erlang distribution is generally defined to have integer values\n        for the shape parameter.  This is not enforced by the `erlang` class.\n        When fitting the distribution, it will generally return a non-integer\n        value for the shape parameter.  By using the keyword argument\n        `f0=<integer>`, the fit method can be constrained to fit the data to\n        a specific integer shape parameter.')
    def fit(self, data, *args, **kwds):
        return super().fit(data, *args, **kwds)