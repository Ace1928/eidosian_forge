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
def _normalize_score_results(scores, scaler_score_key='score'):
    """Creates a scoring dictionary based on the type of `scores`"""
    if isinstance(scores[0], dict):
        return _aggregate_score_dicts(scores)
    return {scaler_score_key: scores}