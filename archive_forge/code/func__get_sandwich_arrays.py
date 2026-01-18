import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def _get_sandwich_arrays(results, cov_type=''):
    """Helper function to get scores from results

    Parameters
    """
    if isinstance(results, tuple):
        jac, hessian_inv = results
        xu = jac = np.asarray(jac)
        hessian_inv = np.asarray(hessian_inv)
    elif hasattr(results, 'model'):
        if hasattr(results, '_results'):
            results = results._results
        if hasattr(results.model, 'jac'):
            xu = results.model.jac(results.params)
            hessian_inv = np.linalg.inv(results.model.hessian(results.params))
        elif hasattr(results.model, 'score_obs'):
            xu = results.model.score_obs(results.params)
            hessian_inv = np.linalg.inv(results.model.hessian(results.params))
        else:
            xu = results.model.wexog * results.wresid[:, None]
            hessian_inv = np.asarray(results.normalized_cov_params)
        if hasattr(results.model, 'freq_weights') and (not cov_type == 'clu'):
            xu /= np.sqrt(np.asarray(results.model.freq_weights)[:, None])
    else:
        raise ValueError('need either tuple of (jac, hessian_inv) or results' + 'instance')
    return (xu, hessian_inv)