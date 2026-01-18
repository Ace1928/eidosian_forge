from abc import abstractmethod
from copy import deepcopy
from math import ceil, floor, log
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context, is_classifier
from ..metrics._scorer import get_scorer_names
from ..utils import resample
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from . import ParameterGrid, ParameterSampler
from ._search import BaseSearchCV
from ._split import _yields_constant_splits, check_cv
def _top_k(results, k, itr):
    iteration, mean_test_score, params = (np.asarray(a) for a in (results['iter'], results['mean_test_score'], results['params']))
    iter_indices = np.flatnonzero(iteration == itr)
    scores = mean_test_score[iter_indices]
    sorted_indices = np.roll(np.argsort(scores), np.count_nonzero(np.isnan(scores)))
    return np.array(params[iter_indices][sorted_indices[-k:]])