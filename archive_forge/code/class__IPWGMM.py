import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
class _IPWGMM(_TEGMMGeneric1):
    """ GMM for aipw treatment effect and potential outcome

    uses unweighted outcome regression
    """

    def momcond(self, params):
        ra = self.teff
        res_select = ra.results_select
        tind = ra.treatment
        endog = ra.model_pool.endog
        effect_group = self.effect_group
        tm = params[:2]
        ps = params[2:]
        prob_sel = np.asarray(res_select.model.predict(ps))
        prob_sel = np.clip(prob_sel, 0.01, 0.99)
        prob = prob_sel
        if effect_group == 'all':
            probt = None
        elif effect_group in [1, 'treated']:
            probt = prob
        elif effect_group in [0, 'untreated', 'control']:
            probt = 1 - prob
        elif isinstance(effect_group, np.ndarray):
            probt = probt
        else:
            raise ValueError('incorrect option for effect_group')
        w = tind / prob + (1 - tind) / (1 - prob)
        if probt is not None:
            w *= probt
        treat_ind = np.column_stack((tind, np.ones(len(tind))))
        mm = (w * (endog - treat_ind.dot(tm)))[:, None] * treat_ind
        mom_select = res_select.model.score_obs(ps)
        moms = np.column_stack((mm, mom_select))
        return moms