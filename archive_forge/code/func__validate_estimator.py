from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
from ..utils import Bunch, _print_elapsed_time, check_random_state
from ..utils._tags import _safe_tags
from ..utils.metaestimators import _BaseComposition
def _validate_estimator(self, default=None):
    """Check the base estimator.

        Sets the `estimator_` attributes.
        """
    if self.estimator is not None:
        self.estimator_ = self.estimator
    else:
        self.estimator_ = default