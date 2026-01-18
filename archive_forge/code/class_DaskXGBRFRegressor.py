import collections
import logging
import platform
import socket
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, update_wrapper
from threading import Thread
from typing import (
import numpy
from . import collective, config
from ._typing import _T, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import DataFrame, LazyLoader, concat, lazy_isinstance
from .core import (
from .data import _is_cudf_ser, _is_cupy_array
from .sklearn import (
from .tracker import RabitTracker, get_host_ip
from .training import train as worker_train
@xgboost_model_doc('Implementation of the Scikit-Learn API for XGBoost Random Forest Regressor.\n\n    .. versionadded:: 1.4.0\n\n', ['model', 'objective'], extra_parameters='\n    n_estimators : int\n        Number of trees in random forest to fit.\n')
class DaskXGBRFRegressor(DaskXGBRegressor):

    @_deprecate_positional_args
    def __init__(self, *, learning_rate: Optional[float]=1, subsample: Optional[float]=0.8, colsample_bynode: Optional[float]=0.8, reg_lambda: Optional[float]=1e-05, **kwargs: Any) -> None:
        super().__init__(learning_rate=learning_rate, subsample=subsample, colsample_bynode=colsample_bynode, reg_lambda=reg_lambda, **kwargs)

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params['num_parallel_tree'] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1

    def fit(self, X: _DataT, y: _DaskCollection, *, sample_weight: Optional[_DaskCollection]=None, base_margin: Optional[_DaskCollection]=None, eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]]=None, eval_metric: Optional[Union[str, Sequence[str], Callable]]=None, early_stopping_rounds: Optional[int]=None, verbose: Union[int, bool]=True, xgb_model: Optional[Union[Booster, XGBModel]]=None, sample_weight_eval_set: Optional[Sequence[_DaskCollection]]=None, base_margin_eval_set: Optional[Sequence[_DaskCollection]]=None, feature_weights: Optional[_DaskCollection]=None, callbacks: Optional[Sequence[TrainingCallback]]=None) -> 'DaskXGBRFRegressor':
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        _check_rf_callback(early_stopping_rounds, callbacks)
        super().fit(**args)
        return self