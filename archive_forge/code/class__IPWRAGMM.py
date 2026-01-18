import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
class _IPWRAGMM(_TEGMMGeneric1):
    """ GMM for ipwra treatment effect and potential outcome
    """

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select
        ppom = params[1]
        mask = np.arange(len(params)) != 1
        params = params[mask]
        k = ra.results0.model.exog.shape[1]
        pm = params[0]
        p0 = params[1:k + 1]
        p1 = params[k + 1:2 * k + 1]
        ps = params[-6:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        exog = ra.exog_grouped
        tind = np.zeros(len(treat_mask))
        tind[-treat_mask.sum():] = 1
        prob_sel = np.asarray(res_select.model.predict(ps))
        prob_sel = np.clip(prob_sel, 0.001, 0.999)
        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]
        effect_group = self.effect_group
        if effect_group == 'all':
            w0 = 1 / (1 - prob0)
            w1 = 1 / prob1
            sind = 1
        elif effect_group in [1, 'treated']:
            w0 = prob0 / (1 - prob0)
            w1 = prob1 / prob1
            sind = tind / tind.mean()
        elif effect_group in [0, 'untreated', 'control']:
            w0 = (1 - prob0) / (1 - prob0)
            w1 = (1 - prob1) / prob1
            sind = 1 - tind
            sind /= sind.mean()
        else:
            raise ValueError('incorrect option for effect_group')
        fitted0 = mod0.predict(p0, exog)
        mom0 = _mom_olsex(p0, model=mod0) * w0[:, None]
        fitted1 = mod1.predict(p1, exog)
        mom1 = _mom_olsex(p1, model=mod1) * w1[:, None]
        mom_outcome = block_diag(mom0, mom1)
        mm = (fitted1 - fitted0 - pm) * sind
        mpom = (fitted0 - ppom) * sind
        mm = np.column_stack((mm, mpom))
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask], mom_select[treat_mask]), axis=0)
        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms