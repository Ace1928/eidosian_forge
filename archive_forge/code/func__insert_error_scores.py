import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
def _insert_error_scores(results, error_score):
    """Insert error in `results` by replacing them inplace with `error_score`.

    This only applies to multimetric scores because `_fit_and_score` will
    handle the single metric case.
    """
    successful_score = None
    failed_indices = []
    for i, result in enumerate(results):
        if result['fit_error'] is not None:
            failed_indices.append(i)
        elif successful_score is None:
            successful_score = result['test_scores']
    if isinstance(successful_score, dict):
        formatted_error = {name: error_score for name in successful_score}
        for i in failed_indices:
            results[i]['test_scores'] = formatted_error.copy()
            if 'train_scores' in results[i]:
                results[i]['train_scores'] = formatted_error.copy()