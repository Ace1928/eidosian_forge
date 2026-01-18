import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
class OneWayMixed:
    """
    Model for
    EM implementation of (repeated measures)
    mixed effects model.

    'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.


    Parameters
    ----------
    units : list of units
       the data for the individual units should be attached to the units
    response, fixed and random : formula expression, called as argument to Formula


    *available results and alias*

    (subject to renaming, and coversion to cached attributes)

    params() -> self.a : coefficient for fixed effects or exog
    cov_params() -> self.Sinv : covariance estimate of fixed effects/exog
    bse() : standard deviation of params

    cov_random -> self.D : estimate of random effects covariance
    params_random_units -> [self.units[...].b] : random coefficient for each unit


    *attributes*

    (others)

    self.m : number of units
    self.p : k_vars_fixed
    self.q : k_vars_random
    self.N : nobs (total)


    Notes
    -----
    Fit returns a result instance, but not all results that use the inherited
    methods have been checked.

    Parameters need to change: drop formula and we require a naming convention for
    the units (currently Y,X,Z). - endog, exog_fe, endog_re ?

    logL does not include constant, e.g. sqrt(pi)
    llf is for MLE not for REML


    convergence criteria for iteration
    Currently convergence in the iterative solver is reached if either the loglikelihood
    *or* the fixed effects parameter do not change above tolerance.

    In some examples, the fixed effects parameters converged to 1e-5 within 150 iterations
    while the log likelihood did not converge within 2000 iterations. This might be
    the case if the fixed effects parameters are well estimated, but there are still
    changes in the random effects. If params_rtol and params_atol are set at a higher
    level, then the random effects might not be estimated to a very high precision.

    The above was with a misspecified model, without a constant. With a
    correctly specified model convergence is fast, within a few iterations
    (6 in example).
    """

    def __init__(self, units):
        self.units = units
        self.m = len(self.units)
        self.n_units = self.m
        self.N = sum((unit.X.shape[0] for unit in self.units))
        self.nobs = self.N
        d = self.units[0].X
        self.p = d.shape[1]
        self.k_exog_fe = self.p
        self.a = np.zeros(self.p, np.float64)
        d = self.units[0].Z
        self.q = d.shape[1]
        self.k_exog_re = self.q
        self.D = np.zeros((self.q,) * 2, np.float64)
        self.sigma = 1.0
        self.dev = np.inf

    def _compute_a(self):
        """fixed effects parameters

        Display (3.1) of
        Laird, Lange, Stram (see help(Mixed)).
        """
        for unit in self.units:
            unit.fit(self.a, self.D, self.sigma)
        S = sum([unit.compute_xtwx() for unit in self.units])
        Y = sum([unit.compute_xtwy() for unit in self.units])
        self.Sinv = L.pinv(S)
        self.a = np.dot(self.Sinv, Y)

    def _compute_sigma(self, ML=False):
        """
        Estimate sigma. If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.6) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.8).

        sigma is the standard deviation of the noise (residual)
        """
        sigmasq = 0.0
        for unit in self.units:
            if ML:
                W = unit.W
            else:
                unit.compute_P(self.Sinv)
                W = unit.P
            t = unit.r - np.dot(unit.Z, unit.b)
            sigmasq += np.power(t, 2).sum()
            sigmasq += self.sigma ** 2 * np.trace(np.identity(unit.n) - self.sigma ** 2 * W)
        self.sigma = np.sqrt(sigmasq / self.N)

    def _compute_D(self, ML=False):
        """
        Estimate random effects covariance D.
        If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.7) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.9).
        """
        D = 0.0
        for unit in self.units:
            if ML:
                W = unit.W
            else:
                unit.compute_P(self.Sinv)
                W = unit.P
            D += np.multiply.outer(unit.b, unit.b)
            t = np.dot(unit.Z, self.D)
            D += self.D - np.dot(np.dot(t.T, W), t)
        self.D = D / self.m

    def cov_fixed(self):
        """
        Approximate covariance of estimates of fixed effects.

        Just after Display (3.10) in Laird, Lange, Stram (see help(Mixed)).
        """
        return self.Sinv

    def cov_random(self):
        """
        Estimate random effects covariance D.

        If ML is True, return the ML estimate of sigma, else return the REML estimate.

        see _compute_D, alias for self.D
        """
        return self.D

    @property
    def params(self):
        """
        estimated coefficients for exogeneous variables or fixed effects

        see _compute_a, alias for self.a
        """
        return self.a

    @property
    def params_random_units(self):
        """random coefficients for each unit

        """
        return np.array([unit.b for unit in self.units])

    def cov_params(self):
        """
        estimated covariance for coefficients for exogeneous variables or fixed effects

        see cov_fixed, and Sinv in _compute_a
        """
        return self.cov_fixed()

    @property
    def bse(self):
        """
        standard errors of estimated coefficients for exogeneous variables (fixed)

        """
        return np.sqrt(np.diag(self.cov_params()))

    def deviance(self, ML=False):
        """deviance defined as 2 times the negative loglikelihood

        """
        return -2 * self.logL(ML=ML)

    def logL(self, ML=False):
        """
        Return log-likelihood, REML by default.
        """
        logL = 0.0
        for unit in self.units:
            logL += unit.logL(a=self.a, ML=ML)
        if not ML:
            logL += np.log(L.det(self.Sinv)) / 2
        return logL

    def initialize(self):
        S = sum([np.dot(unit.X.T, unit.X) for unit in self.units])
        Y = sum([np.dot(unit.X.T, unit.Y) for unit in self.units])
        self.a = L.lstsq(S, Y, rcond=-1)[0]
        D = 0
        t = 0
        sigmasq = 0
        for unit in self.units:
            unit.r = unit.Y - np.dot(unit.X, self.a)
            if self.q > 1:
                unit.b = L.lstsq(unit.Z, unit.r, rcond=-1)[0]
            else:
                Z = unit.Z.reshape((unit.Z.shape[0], 1))
                unit.b = L.lstsq(Z, unit.r, rcond=-1)[0]
            sigmasq += np.power(unit.Y, 2).sum() - (self.a * np.dot(unit.X.T, unit.Y)).sum() - (unit.b * np.dot(unit.Z.T, unit.r)).sum()
            D += np.multiply.outer(unit.b, unit.b)
            t += L.pinv(np.dot(unit.Z.T, unit.Z))
        self.df_resid = self.N - (self.m - 1) * self.q - self.p
        sigmasq /= self.N - (self.m - 1) * self.q - self.p
        self.sigma = np.sqrt(sigmasq)
        self.D = (D - sigmasq * t) / self.m

    def cont(self, ML=False, rtol=1e-05, params_rtol=1e-05, params_atol=0.0001):
        """convergence check for iterative estimation

        """
        self.dev, old = (self.deviance(ML=ML), self.dev)
        self.history['llf'].append(self.dev)
        self.history['params'].append(self.a.copy())
        self.history['D'].append(self.D.copy())
        if np.fabs((self.dev - old) / self.dev) < rtol:
            self.termination = 'llf'
            return False
        if np.all(np.abs(self.a - self._a_old) < params_rtol * self.a + params_atol):
            self.termination = 'params'
            return False
        self._a_old = self.a.copy()
        return True

    def fit(self, maxiter=100, ML=False, rtol=1e-05, params_rtol=1e-06, params_atol=1e-06):
        self._a_old = np.inf * self.a
        self.history = {'llf': [], 'params': [], 'D': []}
        for i in range(maxiter):
            self._compute_a()
            self._compute_sigma(ML=ML)
            self._compute_D(ML=ML)
            if not self.cont(ML=ML, rtol=rtol, params_rtol=params_rtol, params_atol=params_atol):
                break
        else:
            self.termination = 'maxiter'
            print('Warning: maximum number of iterations reached')
        self.iterations = i
        results = OneWayMixedResults(self)
        results.scale = 1
        results.normalized_cov_params = self.cov_params()
        return results