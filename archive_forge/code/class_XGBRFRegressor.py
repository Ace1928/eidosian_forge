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
@xgboost_model_doc('scikit-learn API for XGBoost random forest regression.', ['model', 'objective'], extra_parameters='\n    n_estimators : Optional[int]\n        Number of trees in random forest to fit.\n')
class XGBRFRegressor(XGBRegressor):

    @_deprecate_positional_args
    def __init__(self, *, learning_rate: float=1.0, subsample: float=0.8, colsample_bynode: float=0.8, reg_lambda: float=1e-05, **kwargs: Any) -> None:
        super().__init__(learning_rate=learning_rate, subsample=subsample, colsample_bynode=colsample_bynode, reg_lambda=reg_lambda, **kwargs)
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params['num_parallel_tree'] = super().get_num_boosting_rounds()
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1

    @_deprecate_positional_args
    def fit(self, X: ArrayLike, y: ArrayLike, *, sample_weight: Optional[ArrayLike]=None, base_margin: Optional[ArrayLike]=None, eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]]=None, eval_metric: Optional[Union[str, Sequence[str], Metric]]=None, early_stopping_rounds: Optional[int]=None, verbose: Optional[Union[bool, int]]=True, xgb_model: Optional[Union[Booster, str, XGBModel]]=None, sample_weight_eval_set: Optional[Sequence[ArrayLike]]=None, base_margin_eval_set: Optional[Sequence[ArrayLike]]=None, feature_weights: Optional[ArrayLike]=None, callbacks: Optional[Sequence[TrainingCallback]]=None) -> 'XGBRFRegressor':
        args = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        _check_rf_callback(early_stopping_rounds, callbacks)
        super().fit(**args)
        return self