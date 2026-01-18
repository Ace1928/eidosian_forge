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
def _check_input_parameters(self, X, y, split_params):
    if not _yields_constant_splits(self._checked_cv_orig):
        raise ValueError('The cv parameter must yield consistent folds across calls to split(). Set its random_state to an int, or set shuffle=False.')
    if self.resource != 'n_samples' and self.resource not in self.estimator.get_params():
        raise ValueError(f'Cannot use resource={self.resource} which is not supported by estimator {self.estimator.__class__.__name__}')
    if isinstance(self, HalvingRandomSearchCV):
        if self.min_resources == self.n_candidates == 'exhaust':
            raise ValueError("n_candidates and min_resources cannot be both set to 'exhaust'.")
    self.min_resources_ = self.min_resources
    if self.min_resources_ in ('smallest', 'exhaust'):
        if self.resource == 'n_samples':
            n_splits = self._checked_cv_orig.get_n_splits(X, y, **split_params)
            magic_factor = 2
            self.min_resources_ = n_splits * magic_factor
            if is_classifier(self.estimator):
                y = self._validate_data(X='no_validation', y=y)
                check_classification_targets(y)
                n_classes = np.unique(y).shape[0]
                self.min_resources_ *= n_classes
        else:
            self.min_resources_ = 1
    self.max_resources_ = self.max_resources
    if self.max_resources_ == 'auto':
        if not self.resource == 'n_samples':
            raise ValueError("resource can only be 'n_samples' when max_resources='auto'")
        self.max_resources_ = _num_samples(X)
    if self.min_resources_ > self.max_resources_:
        raise ValueError(f'min_resources_={self.min_resources_} is greater than max_resources_={self.max_resources_}.')
    if self.min_resources_ == 0:
        raise ValueError(f'min_resources_={self.min_resources_}: you might have passed an empty dataset X.')