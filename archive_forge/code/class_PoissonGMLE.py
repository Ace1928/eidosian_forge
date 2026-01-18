import numpy as np
from scipy import stats
from scipy.special import factorial
from statsmodels.base.model import GenericLikelihoodModel
class PoissonGMLE(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    """

    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        The log likelihood of the model evaluated at `params`

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        XB = np.dot(self.exog, params)
        endog = self.endog
        return np.exp(XB) - endog * XB + np.log(factorial(endog))

    def predict_distribution(self, exog):
        """return frozen scipy.stats distribution with mu at estimated prediction
        """
        if not hasattr(self, 'result'):
            raise ValueError
        else:
            result = self.result
            params = result.params
            mu = np.exp(np.dot(exog, params))
            return stats.poisson(mu, loc=0)