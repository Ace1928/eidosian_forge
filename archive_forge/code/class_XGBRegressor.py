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
@xgboost_model_doc('Implementation of the scikit-learn API for XGBoost regression.', ['estimators', 'model', 'objective'])
class XGBRegressor(XGBModel, XGBRegressorBase):

    @_deprecate_positional_args
    def __init__(self, *, objective: SklObjective='reg:squarederror', **kwargs: Any) -> None:
        super().__init__(objective=objective, **kwargs)