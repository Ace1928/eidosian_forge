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
def _check_rf_callback(early_stopping_rounds: Optional[int], callbacks: Optional[Sequence[TrainingCallback]]) -> None:
    if early_stopping_rounds is not None or callbacks is not None:
        raise NotImplementedError('`early_stopping_rounds` and `callbacks` are not implemented for random forest.')