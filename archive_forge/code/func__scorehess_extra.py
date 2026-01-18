import numpy as np
from scipy import stats
def _scorehess_extra(self, params=None, exog_extra=None, exog2_extra=None, hess_kwds=None):
    """Experimental helper function for variable addition score test.

    This uses score and hessian factor at the params which should be the
    params of the restricted model.

    """
    if hess_kwds is None:
        hess_kwds = {}
    model = self.model
    if params is None:
        params = self.params
    exog_o1, exog_o2 = model._get_exogs()
    if exog_o2 is None:
        exog_o2 = np.ones((exog_o1.shape[0], 1))
    k_mean = exog_o1.shape[1]
    k_prec = exog_o2.shape[1]
    if exog_extra is not None:
        exog = np.column_stack((exog_o1, exog_extra))
    else:
        exog = exog_o1
    if exog2_extra is not None:
        exog2 = np.column_stack((exog_o2, exog2_extra))
    else:
        exog2 = exog_o2
    k_mean_new = exog.shape[1]
    k_prec_new = exog2.shape[1]
    k_cm = k_mean_new - k_mean
    k_cp = k_prec_new - k_prec
    k_constraints = k_cm + k_cp
    index_mean = np.arange(k_mean, k_mean_new)
    index_prec = np.arange(k_mean_new + k_prec, k_mean_new + k_prec_new)
    r_matrix = np.zeros((k_constraints, len(params) + k_constraints))
    r_matrix[:k_cm, index_mean] = np.eye(k_cm)
    r_matrix[k_cm:k_cm + k_cp, index_prec] = np.eye(k_cp)
    if hasattr(model, 'score_hessian_factor'):
        sf, hf = model.score_hessian_factor(params, return_hessian=True, **hess_kwds)
    else:
        sf = model.score_factor(params)
        hf = model.hessian_factor(params, **hess_kwds)
    sf1, sf2 = sf
    hf11, hf12, hf22 = hf
    d1 = sf1[:, None] * exog
    d2 = sf2[:, None] * exog2
    score_obs = np.column_stack((d1, d2))
    d11 = (exog.T * hf11).dot(exog)
    d12 = (exog.T * hf12).dot(exog2)
    d22 = (exog2.T * hf22).dot(exog2)
    hessian = np.block([[d11, d12], [d12.T, d22]])
    return (score_obs, hessian, k_constraints, r_matrix)