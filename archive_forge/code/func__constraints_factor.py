import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def _constraints_factor(encoding_matrix, comparison='pairwise', k_params=None, idx_start=None):
    """helper function to create constraints based on encoding matrix

    Parameters
    ----------
    encoding_matrix : ndarray
        contrast matrix for the encoding of a factor as defined by patsy.
        The number of rows should be equal to the number of levels or categories
        of the factor, the number of columns should be equal to the number
        of parameters for this factor.
    comparison : str
        Currently only 'pairwise' is implemented. The restriction matrix
        can be used for testing the hypothesis that all pairwise differences
        are zero.
    k_params : int
        number of parameters
    idx_start : int
        Index of the first parameter of this factor. The restrictions on the
        factor are inserted as a block in the full restriction matrix starting
        at column with index `idx_start`.

    Returns
    -------
    contrast : ndarray
        Contrast or restriction matrix that can be used in hypothesis test
        of model results. The number of columns is k_params.
    """
    cm = encoding_matrix
    k_level, k_p = cm.shape
    import statsmodels.sandbox.stats.multicomp as mc
    if comparison in ['pairwise', 'pw', 'pairs']:
        c_all = -mc.contrast_allpairs(k_level)
    else:
        raise NotImplementedError('currentlyonly pairwise comparison')
    contrasts = c_all.dot(cm)
    if k_params is not None:
        if idx_start is None:
            raise ValueError('if k_params is not None, then idx_start is required')
        contrasts = _embed_constraints(contrasts, k_params, idx_start)
    return contrasts