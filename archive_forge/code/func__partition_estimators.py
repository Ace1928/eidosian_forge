from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
from ..utils import Bunch, _print_elapsed_time, check_random_state
from ..utils._tags import _safe_tags
from ..utils.metaestimators import _BaseComposition
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)
    return (n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist())