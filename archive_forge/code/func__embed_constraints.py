import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def _embed_constraints(contrasts, k_params, idx_start, index=None):
    """helper function to expand constraints to a full restriction matrix

    Parameters
    ----------
    contrasts : ndarray
        restriction matrix for t_test
    k_params : int
        number of parameters
    idx_start : int
        Index of the first parameter of this factor. The restrictions on the
        factor are inserted as a block in the full restriction matrix starting
        at column with index `idx_start`.
    index : slice or ndarray
        Column index if constraints do not form a block in the full restriction
        matrix, i.e. if parameters that are subject to restrictions are not
        consecutive in the list of parameters.
        If index is not None, then idx_start is ignored.

    Returns
    -------
    contrasts : ndarray
        restriction matrix with k_params columns and number of rows equal to
        the number of restrictions.
    """
    k_c, k_p = contrasts.shape
    c = np.zeros((k_c, k_params))
    if index is None:
        c[:, idx_start:idx_start + k_p] = contrasts
    else:
        c[:, index] = contrasts
    return c