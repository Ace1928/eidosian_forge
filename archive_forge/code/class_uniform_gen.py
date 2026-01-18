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
class uniform_gen(rv_continuous):
    """A uniform continuous random variable.

    In the standard form, the distribution is uniform on ``[0, 1]``. Using
    the parameters ``loc`` and ``scale``, one obtains the uniform distribution
    on ``[loc, loc + scale]``.

    %(before_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return random_state.uniform(0.0, 1.0, size)

    def _pdf(self, x):
        return 1.0 * (x == x)

    def _cdf(self, x):
        return x

    def _ppf(self, q):
        return q

    def _stats(self):
        return (0.5, 1.0 / 12, 0, -1.2)

    def _entropy(self):
        return 0.0

    @_call_super_mom
    def fit(self, data, *args, **kwds):
        """
        Maximum likelihood estimate for the location and scale parameters.

        `uniform.fit` uses only the following parameters.  Because exact
        formulas are used, the parameters related to optimization that are
        available in the `fit` method of other distributions are ignored
        here.  The only positional argument accepted is `data`.

        Parameters
        ----------
        data : array_like
            Data to use in calculating the maximum likelihood estimate.
        floc : float, optional
            Hold the location parameter fixed to the specified value.
        fscale : float, optional
            Hold the scale parameter fixed to the specified value.

        Returns
        -------
        loc, scale : float
            Maximum likelihood estimates for the location and scale.

        Notes
        -----
        An error is raised if `floc` is given and any values in `data` are
        less than `floc`, or if `fscale` is given and `fscale` is less
        than ``data.max() - data.min()``.  An error is also raised if both
        `floc` and `fscale` are given.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import uniform

        We'll fit the uniform distribution to `x`:

        >>> x = np.array([2, 2.5, 3.1, 9.5, 13.0])

        For a uniform distribution MLE, the location is the minimum of the
        data, and the scale is the maximum minus the minimum.

        >>> loc, scale = uniform.fit(x)
        >>> loc
        2.0
        >>> scale
        11.0

        If we know the data comes from a uniform distribution where the support
        starts at 0, we can use `floc=0`:

        >>> loc, scale = uniform.fit(x, floc=0)
        >>> loc
        0.0
        >>> scale
        13.0

        Alternatively, if we know the length of the support is 12, we can use
        `fscale=12`:

        >>> loc, scale = uniform.fit(x, fscale=12)
        >>> loc
        1.5
        >>> scale
        12.0

        In that last example, the support interval is [1.5, 13.5].  This
        solution is not unique.  For example, the distribution with ``loc=2``
        and ``scale=12`` has the same likelihood as the one above.  When
        `fscale` is given and it is larger than ``data.max() - data.min()``,
        the parameters returned by the `fit` method center the support over
        the interval ``[data.min(), data.max()]``.

        """
        if len(args) > 0:
            raise TypeError('Too many arguments.')
        floc = kwds.pop('floc', None)
        fscale = kwds.pop('fscale', None)
        _remove_optimizer_parameters(kwds)
        if floc is not None and fscale is not None:
            raise ValueError('All parameters fixed. There is nothing to optimize.')
        data = np.asarray(data)
        if not np.isfinite(data).all():
            raise ValueError('The data contains non-finite values.')
        if fscale is None:
            if floc is None:
                loc = data.min()
                scale = np.ptp(data)
            else:
                loc = floc
                scale = data.max() - loc
                if data.min() < loc:
                    raise FitDataError('uniform', lower=loc, upper=loc + scale)
        else:
            ptp = np.ptp(data)
            if ptp > fscale:
                raise FitUniformFixedScaleDataError(ptp=ptp, fscale=fscale)
            loc = data.min() - 0.5 * (fscale - ptp)
            scale = fscale
        return (float(loc), float(scale))