from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
class rv_sample(rv_discrete):
    """A 'sample' discrete distribution defined by the support and values.

    The ctor ignores most of the arguments, only needs the `values` argument.
    """

    def __init__(self, a=0, b=inf, name=None, badvalue=None, moment_tol=1e-08, values=None, inc=1, longname=None, shapes=None, seed=None):
        super(rv_discrete, self).__init__(seed)
        if values is None:
            raise ValueError('rv_sample.__init__(..., values=None,...)')
        self._ctor_param = dict(a=a, b=b, name=name, badvalue=badvalue, moment_tol=moment_tol, values=values, inc=inc, longname=longname, shapes=shapes, seed=seed)
        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        self.vecentropy = self._entropy
        xk, pk = values
        if np.shape(xk) != np.shape(pk):
            raise ValueError('xk and pk must have the same shape.')
        if np.less(pk, 0.0).any():
            raise ValueError('All elements of pk must be non-negative.')
        if not np.allclose(np.sum(pk), 1):
            raise ValueError('The sum of provided pk is not 1.')
        if not len(set(np.ravel(xk))) == np.size(xk):
            raise ValueError('xk may not contain duplicate values.')
        indx = np.argsort(np.ravel(xk))
        self.xk = np.take(np.ravel(xk), indx, 0)
        self.pk = np.take(np.ravel(pk), indx, 0)
        self.a = self.xk[0]
        self.b = self.xk[-1]
        self.qvals = np.cumsum(self.pk, axis=0)
        self.shapes = ' '
        self._construct_argparser(meths_to_inspect=[self._pmf], locscale_in='loc=0', locscale_out='loc, 1')
        self._attach_methods()
        self._construct_docstrings(name, longname)

    def __getstate__(self):
        dct = self.__dict__.copy()
        attrs = ['_parse_args', '_parse_args_stats', '_parse_args_rvs']
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        """Attaches dynamically created argparser methods."""
        self._attach_argparser_methods()

    def _get_support(self, *args):
        """Return the support of the (unscaled, unshifted) distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support.
        """
        return (self.a, self.b)

    def _pmf(self, x):
        return np.select([x == k for k in self.xk], [np.broadcast_arrays(p, x)[0] for p in self.pk], 0)

    def _cdf(self, x):
        xx, xxk = np.broadcast_arrays(x[:, None], self.xk)
        indx = np.argmax(xxk > xx, axis=-1) - 1
        return self.qvals[indx]

    def _ppf(self, q):
        qq, sqq = np.broadcast_arrays(q[..., None], self.qvals)
        indx = argmax(sqq >= qq, axis=-1)
        return self.xk[indx]

    def _rvs(self, size=None, random_state=None):
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self._ppf(U)[0]
        else:
            Y = self._ppf(U)
        return Y

    def _entropy(self):
        return stats.entropy(self.pk)

    def generic_moment(self, n):
        n = asarray(n)
        return np.sum(self.xk ** n[np.newaxis, ...] * self.pk, axis=0)

    def _expect(self, fun, lb, ub, *args, **kwds):
        supp = self.xk[(lb <= self.xk) & (self.xk <= ub)]
        vals = fun(supp)
        return np.sum(vals)