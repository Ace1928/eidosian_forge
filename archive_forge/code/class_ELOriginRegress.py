import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.tools.tools import add_constant
class ELOriginRegress:
    """
    Empirical Likelihood inference and estimation for linear regression
    through the origin.

    Parameters
    ----------
    endog: nx1 array
        Array of response variables.

    exog: nxk array
        Array of exogenous variables.  Assumes no array of ones

    Attributes
    ----------
    endog : nx1 array
        Array of response variables

    exog : nxk array
        Array of exogenous variables.  Assumes no array of ones.

    nobs : float
        Number of observations.

    nvar : float
        Number of exogenous regressors.
    """

    def __init__(self, endog, exog):
        self.endog = endog
        self.exog = exog
        self.nobs = self.exog.shape[0]
        try:
            self.nvar = float(exog.shape[1])
        except IndexError:
            self.nvar = 1.0

    def fit(self):
        """
        Fits the model and provides regression results.

        Returns
        -------
        Results : class
            Empirical likelihood regression class.
        """
        exog_with = add_constant(self.exog, prepend=True)
        restricted_model = OLS(self.endog, exog_with)
        restricted_fit = restricted_model.fit()
        restricted_el = restricted_fit.el_test(np.array([0]), np.array([0]), ret_params=1)
        params = np.squeeze(restricted_el[3])
        beta_hat_llr = restricted_el[0]
        llf = np.sum(np.log(restricted_el[2]))
        return OriginResults(restricted_model, params, beta_hat_llr, llf)

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(add_constant(exog, prepend=True), params)