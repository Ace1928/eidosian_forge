import numpy as np
from scipy import stats
from scipy.special import factorial
from statsmodels.base.model import GenericLikelihoodModel
class PoissonZiGMLE(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same statistical model
    as discretemod.Poisson but adds offset and zero-inflation.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    There are numerical problems if there is no zero-inflation.

    """

    def __init__(self, endog, exog=None, offset=None, missing='none', **kwds):
        self.k_extra = 1
        super().__init__(endog, exog, missing=missing, extra_params_names=['zi'], **kwds)
        if offset is not None:
            if offset.ndim == 1:
                offset = offset[:, None]
            self.offset = offset.ravel()
        else:
            self.offset = 0.0
        if exog is None:
            self.exog = np.ones((self.nobs, 1))
        self.nparams = self.exog.shape[1]
        self.start_params = np.hstack((np.ones(self.nparams), 0))
        self.nparams += 1
        self.cloneattr = ['start_params']

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
        beta = params[:-1]
        gamm = 1 / (1 + np.exp(params[-1]))
        XB = self.offset + np.dot(self.exog, beta)
        endog = self.endog
        nloglik = -np.log(1 - gamm) + np.exp(XB) - endog * XB + np.log(factorial(endog))
        nloglik[endog == 0] = -np.log(gamm + np.exp(-nloglik[endog == 0]))
        return nloglik