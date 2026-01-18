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
def _get_datasets(self) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    pd_datasets = {}
    for key, ray_dataset in self.datasets.items():
        pd_dataset = ray_dataset.to_pandas(limit=float('inf'))
        if self.label_column:
            pd_datasets[key] = (pd_dataset.drop(self.label_column, axis=1), pd_dataset[self.label_column])
        else:
            pd_datasets[key] = (pd_dataset, None)
    return pd_datasets