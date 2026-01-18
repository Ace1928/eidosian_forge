import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
def _load_model_attributes(self, config: dict) -> None:
    """Load model attributes without hyper-parameters."""
    from sklearn.base import is_classifier
    booster = self.get_booster()
    self.objective = config['learner']['objective']['name']
    self.booster = config['learner']['gradient_booster']['name']
    self.base_score = config['learner']['learner_model_param']['base_score']
    self.feature_types = booster.feature_types
    if is_classifier(self):
        self.n_classes_ = int(config['learner']['learner_model_param']['num_class'])
        self.n_classes_ = 2 if self.n_classes_ < 2 else self.n_classes_