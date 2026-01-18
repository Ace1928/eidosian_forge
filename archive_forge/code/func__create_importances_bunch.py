import numbers
import numpy as np
from ..ensemble._bagging import _generate_indices
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..model_selection._validation import _aggregate_score_dicts
from ..utils import Bunch, _safe_indexing, check_array, check_random_state
from ..utils._param_validation import (
from ..utils.parallel import Parallel, delayed
def _create_importances_bunch(baseline_score, permuted_score):
    """Compute the importances as the decrease in score.

    Parameters
    ----------
    baseline_score : ndarray of shape (n_features,)
        The baseline score without permutation.
    permuted_score : ndarray of shape (n_features, n_repeats)
        The permuted scores for the `n` repetitions.

    Returns
    -------
    importances : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.
    """
    importances = baseline_score - permuted_score
    return Bunch(importances_mean=np.mean(importances, axis=1), importances_std=np.std(importances, axis=1), importances=importances)