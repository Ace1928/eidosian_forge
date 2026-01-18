import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
class SmoothingSpline(BSpline):
    penmax = 30.0
    method = 'target_df'
    target_df = 5
    default_pen = 0.001
    optimize = True
    '\n    A smoothing spline, which can be used to smooth scatterplots, i.e.\n    a list of (x,y) tuples.\n\n    See fit method for more information.\n\n    '

    def fit(self, y, x=None, weights=None, pen=0.0):
        """
        Fit the smoothing spline to a set of (x,y) pairs.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           weights -- optional array of weights
           pen     -- constant in front of Gram matrix

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.

        ALGORITHM:
           Formally, this solves a minimization:

           fhat = ARGMIN_f SUM_i=1^n (y_i-f(x_i))^2 + pen * int f^(2)^2

           int is integral. pen is lambda (from Hastie)

           See Chapter 5 of

           Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
           Learning." Springer-Verlag. 536 pages.

           for more details.

        TODO:
           Should add arbitrary derivative penalty instead of just
           second derivative.
        """
        banded = True
        if x is None:
            x = self._x
            bt = self._basisx.copy()
        else:
            bt = self.basis(x)
        if pen == 0.0:
            banded = False
        if x.shape != y.shape:
            raise ValueError("x and y shape do not agree, by default x are                the Bspline's internal knots")
        if pen >= self.penmax:
            pen = self.penmax
        if weights is not None:
            self.weights = weights
        else:
            self.weights = 1.0
        _w = np.sqrt(self.weights)
        bt *= _w
        mask = np.flatnonzero(1 - np.all(np.equal(bt, 0), axis=0))
        bt = bt[:, mask]
        y = y[mask]
        self.df_total = y.shape[0]
        bty = np.squeeze(np.dot(bt, _w * y))
        self.N = y.shape[0]
        if not banded:
            self.btb = np.dot(bt, bt.T)
            _g = _band2array(self.g, lower=1, symmetric=True)
            self.coef, _, self.rank = L.lstsq(self.btb + pen * _g, bty)[0:3]
            self.rank = min(self.rank, self.btb.shape[0])
            del _g
        else:
            self.btb = np.zeros(self.g.shape, np.float64)
            nband, nbasis = self.g.shape
            for i in range(nbasis):
                for k in range(min(nband, nbasis - i)):
                    self.btb[k, i] = (bt[i] * bt[i + k]).sum()
            bty.shape = (1, bty.shape[0])
            self.pen = pen
            self.chol, self.coef = solveh_banded(self.btb + pen * self.g, bty, lower=1)
        self.coef = np.squeeze(self.coef)
        self.resid = y * self.weights - np.dot(self.coef, bt)
        self.pen = pen
        del bty
        del mask
        del bt

    def smooth(self, y, x=None, weights=None):
        if self.method == 'target_df':
            if hasattr(self, 'pen'):
                self.fit(y, x=x, weights=weights, pen=self.pen)
            else:
                self.fit_target_df(y, x=x, weights=weights, df=self.target_df)
        elif self.method == 'optimize_gcv':
            self.fit_optimize_gcv(y, x=x, weights=weights)

    def gcv(self):
        """
        Generalized cross-validation score of current fit.

        Craven, P. and Wahba, G.  "Smoothing noisy data with spline functions.
        Estimating the correct degree of smoothing by
        the method of generalized cross-validation."
        Numerische Mathematik, 31(4), 377-403.
        """
        norm_resid = (self.resid ** 2).sum()
        return norm_resid / (self.df_total - self.trace())

    def df_resid(self):
        """
        Residual degrees of freedom in the fit.

        self.N - self.trace()

        where self.N is the number of observations of last fit.
        """
        return self.N - self.trace()

    def df_fit(self):
        """
        How many degrees of freedom used in the fit?

        self.trace()
        """
        return self.trace()

    def trace(self):
        """
        Trace of the smoothing matrix S(pen)

        TODO: addin a reference to Wahba, and whoever else I used.
        """
        if self.pen > 0:
            _invband = _hbspline.invband(self.chol.copy())
            tr = _trace_symbanded(_invband, self.btb, lower=1)
            return tr
        else:
            return self.rank

    def fit_target_df(self, y, x=None, df=None, weights=None, tol=0.001, apen=0, bpen=0.001):
        """
        Fit smoothing spline with approximately df degrees of freedom
        used in the fit, i.e. so that self.trace() is approximately df.

        Uses binary search strategy.

        In general, df must be greater than the dimension of the null space
        of the Gram inner product. For cubic smoothing splines, this means
        that df > 2.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           apen    -- lower bound of penalty for binary search
           bpen    -- upper bound of penalty for binary search

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        """
        df = df or self.target_df
        olddf = y.shape[0] - self.m
        if hasattr(self, 'pen'):
            self.fit(y, x=x, weights=weights, pen=self.pen)
            curdf = self.trace()
            if np.fabs(curdf - df) / df < tol:
                return
            if curdf > df:
                apen, bpen = (self.pen, 2 * self.pen)
            else:
                apen, bpen = (0.0, self.pen)
        while True:
            curpen = 0.5 * (apen + bpen)
            self.fit(y, x=x, weights=weights, pen=curpen)
            curdf = self.trace()
            if curdf > df:
                apen, bpen = (curpen, 2 * curpen)
            else:
                apen, bpen = (apen, curpen)
            if apen >= self.penmax:
                raise ValueError('penalty too large, try setting penmax                    higher or decreasing df')
            if np.fabs(curdf - df) / df < tol:
                break

    def fit_optimize_gcv(self, y, x=None, weights=None, tol=0.001, brack=(-100, 20)):
        """
        Fit smoothing spline trying to optimize GCV.

        Try to find a bracketing interval for scipy.optimize.golden
        based on bracket.

        It is probably best to use target_df instead, as it is
        sometimes difficult to find a bracketing interval.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           brack   -- an initial guess at the bracketing interval

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        """

        def _gcv(pen, y, x):
            self.fit(y, x=x, pen=np.exp(pen))
            a = self.gcv()
            return a
        a = golden(_gcv, args=(y, x), brack=brack, tol=tol)