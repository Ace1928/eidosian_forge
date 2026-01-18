import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..exceptions import NotFittedError
from ..metrics import check_scoring
from ..metrics._scorer import (
from ..utils import Bunch, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_method_params, check_is_fitted, indexable
from ._split import check_cv
from ._validation import (
@staticmethod
def _select_best_index(refit, refit_metric, results):
    """Select index of the best combination of hyperparemeters."""
    if callable(refit):
        best_index = refit(results)
        if not isinstance(best_index, numbers.Integral):
            raise TypeError('best_index_ returned is not an integer')
        if best_index < 0 or best_index >= len(results['params']):
            raise IndexError('best_index_ index out of range')
    else:
        best_index = results[f'rank_test_{refit_metric}'].argmin()
    return best_index