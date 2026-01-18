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
def _generate_candidate_params(self):
    n_candidates_first_iter = self.n_candidates
    if n_candidates_first_iter == 'exhaust':
        n_candidates_first_iter = self.max_resources_ // self.min_resources_
    return ParameterSampler(self.param_distributions, n_candidates_first_iter, random_state=self.random_state)