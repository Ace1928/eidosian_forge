import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize
class TryNCLogit:
    """
    Nested Conditional Logit (RUNMNL), data handling test

    unfinished, does not do anything yet

    """

    def __init__(self, endog, exog_bychoices, ncommon):
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        self.nobs, self.nchoices = endog.shape
        self.nchoices = len(exog_bychoices)
        betaind = [exog_bychoices[ii].shape[1] - ncommon for ii in range(4)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        beta_indices = [np.r_[np.array([0, 1]), z[zi[ii]:zi[ii + 1]]] for ii in range(len(zi) - 1)]
        self.beta_indices = beta_indices
        beta = np.arange(7)
        betaidx_bychoices = [beta[idx] for idx in beta_indices]

    def xbetas(self, params):
        """these are the V_i
        """
        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:, choiceind] = np.dot(self.exog_bychoices[choiceind], params[self.beta_indices[choiceind]])
        return res

    def loglike_leafbranch(self, params, tau):
        xb = self.xbetas(params)
        expxb = np.exp(xb / tau)
        sumexpxb = expxb.sum(1)
        logsumexpxb = np.log(sumexpxb)
        probs = expxb / sumexpxb[:, None]
        return (probs, logsumexpxp)

    def loglike_branch(self, params, tau):
        ivs = []
        for b in branches:
            probs, iv = self.loglike_leafbranch(params, tau)
            ivs.append(iv)
        ivs = np.column_stack(ivs)
        exptiv = np.exp(tau * ivs)
        sumexptiv = exptiv.sum(1)
        logsumexpxb = np.log(sumexpxb)
        probs = exptiv / sumexptiv[:, None]