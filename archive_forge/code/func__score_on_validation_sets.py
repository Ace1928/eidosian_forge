import logging
import os
import tempfile
import warnings
from collections import defaultdict
from time import time
from traceback import format_exc
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.model_selection._validation import _check_multimetric_scoring, _score
import ray.cloudpickle as cpickle
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.constants import TRAIN_DATASET_KEY
from ray.train.sklearn import SklearnCheckpoint
from ray.train.sklearn._sklearn_utils import _has_cpu_params, _set_cpu_params
from ray.train.trainer import BaseTrainer, GenDataset
from ray.util import PublicAPI
from ray.util.joblib import register_ray
def _score_on_validation_sets(self, estimator: BaseEstimator, datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, Dict[str, Any]]:
    results = defaultdict(dict)
    if not datasets:
        return results
    if callable(self.scoring):
        scorers = self.scoring
    elif self.scoring is None or isinstance(self.scoring, str):
        scorers = check_scoring(estimator, self.scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, self.scoring)
    for key, X_y_tuple in datasets.items():
        X_test, y_test = X_y_tuple
        start_time = time()
        try:
            test_scores = _score(estimator, X_test, y_test, scorers)
        except Exception:
            if isinstance(scorers, dict):
                test_scores = {k: np.nan for k in scorers}
            else:
                test_scores = np.nan
            warnings.warn(f'Scoring on validation set {key} failed. The score(s) for this set will be set to nan. Details: \n{format_exc()}', UserWarning)
        score_time = time() - start_time
        results[key]['score_time'] = score_time
        if not isinstance(test_scores, dict):
            test_scores = {'score': test_scores}
        for name in test_scores:
            results[key][f'test_{name}'] = test_scores[name]
    return results