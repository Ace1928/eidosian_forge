from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class NonlinearIVGMM(IVGMM):
    """
    Class for non-linear instrumental variables estimation using GMM

    The model is assumed to have the following moment condition

        E[ z * (y - f(X, beta)] = 0

    Where `y` is the dependent endogenous variable, `x` are the explanatory
    variables and `z` are the instruments. Variables in `x` that are exogenous
    need also be included in z. `f` is a nonlinear function.

    Notation Warning: our name `exog` stands for the explanatory variables,
    and includes both exogenous and explanatory variables that are endogenous,
    i.e. included endogenous variables

    Parameters
    ----------
    endog : array_like
        dependent endogenous variable
    exog : array_like
        explanatory, right hand side variables, including explanatory variables
        that are endogenous.
    instruments : array_like
        Instrumental variables, variables that are exogenous to the error
        in the linear model containing both included and excluded exogenous
        variables
    func : callable
        function for the mean or conditional expectation of the endogenous
        variable. The function will be called with parameters and the array of
        explanatory, right hand side variables, `func(params, exog)`

    Notes
    -----
    This class uses numerical differences to obtain the derivative of the
    objective function. If the jacobian of the conditional mean function, `func`
    is available, then it can be used by subclassing this class and defining
    a method `jac_func`.

    TODO: check required signature of jac_error and jac_func
    """

    def fitstart(self):
        return np.zeros(self.exog.shape[1])

    def __init__(self, endog, exog, instrument, func, **kwds):
        self.func = func
        super().__init__(endog, exog, instrument, **kwds)

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return self.func(params, exog)

    def jac_func(self, params, weights, args=None, centered=True, epsilon=None):
        deriv = approx_fprime(params, self.func, args=(self.exog,), centered=centered, epsilon=epsilon)
        return deriv

    def jac_error(self, params, weights, args=None, centered=True, epsilon=None):
        jac_func = self.jac_func(params, weights, args=None, centered=True, epsilon=None)
        return -jac_func

    def score(self, params, weights, **kwds):
        z = self.instrument
        nobs = z.shape[0]
        jac_u = self.jac_error(params, weights, args=None, epsilon=None, centered=True)
        x = -jac_u
        u = self.get_error(params)
        score = -2 * np.dot(np.dot(x.T, z), weights).dot(np.dot(z.T, u))
        score /= nobs * nobs
        return score