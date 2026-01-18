import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
class TwoPeakLLHNoExog(GenericLikelihoodModel):
    """Fit height of signal peak over background."""
    start_params = [10, 1000]
    cloneattr = ['start_params', 'signal', 'background']
    exog_names = ['n_signal', 'n_background']
    endog_names = ['alpha']

    def __init__(self, endog, exog=None, signal=None, background=None, *args, **kwargs):
        self.signal = signal
        self.background = background
        super().__init__(*args, endog=endog, exog=exog, extra_params_names=self.exog_names, **kwargs)

    def loglike(self, params):
        return -self.nloglike(params)

    def nloglike(self, params):
        endog = self.endog
        return self.nlnlike(params, endog)

    def nlnlike(self, params, endog):
        n_sig = params[0]
        n_bkg = params[1]
        if n_sig < 0 or n_bkg < 0:
            return np.inf
        n_tot = n_bkg + n_sig
        alpha = endog
        sig = self.signal.pdf(alpha)
        bkg = self.background.pdf(alpha)
        sumlogl = np.sum(np.log(n_sig * sig + n_bkg * bkg))
        sumlogl -= n_tot
        return -sumlogl