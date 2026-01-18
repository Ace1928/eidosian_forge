import numbers
import numpy as np
from ..ensemble._bagging import _generate_indices
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..model_selection._validation import _aggregate_score_dicts
from ..utils import Bunch, _safe_indexing, check_array, check_random_state
from ..utils._param_validation import (
from ..utils.parallel import Parallel, delayed
def _weights_scorer(scorer, estimator, X, y, sample_weight):
    if sample_weight is not None:
        return scorer(estimator, X, y, sample_weight=sample_weight)
    return scorer(estimator, X, y)