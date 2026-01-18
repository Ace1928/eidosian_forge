import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def _get_pairs_labels(k_level, level_names):
    """helper function for labels for pairwise comparisons
    """
    idx_pairs_all = np.triu_indices(k_level, 1)
    labels = ['{}-{}'.format(level_names[name[1]], level_names[name[0]]) for name in zip(*idx_pairs_all)]
    return labels