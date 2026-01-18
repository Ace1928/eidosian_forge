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
@xgboost_model_doc('Implementation of the Scikit-Learn API for XGBoost Ranking.\n\n    .. versionadded:: 1.4.0\n\n', ['estimators', 'model'], end_note='\n        .. note::\n\n            For dask implementation, group is not supported, use qid instead.\n')
class DaskXGBRanker(DaskScikitLearnBase, XGBRankerMixIn):

    @_deprecate_positional_args
    def __init__(self, *, objective: str='rank:pairwise', **kwargs: Any):
        if callable(objective):
            raise ValueError('Custom objective function not supported by XGBRanker.')
        super().__init__(objective=objective, kwargs=kwargs)

    async def _fit_async(self, X: _DataT, y: _DaskCollection, group: Optional[_DaskCollection], qid: Optional[_DaskCollection], sample_weight: Optional[_DaskCollection], base_margin: Optional[_DaskCollection], eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]], sample_weight_eval_set: Optional[Sequence[_DaskCollection]], base_margin_eval_set: Optional[Sequence[_DaskCollection]], eval_group: Optional[Sequence[_DaskCollection]], eval_qid: Optional[Sequence[_DaskCollection]], eval_metric: Optional[Union[str, Sequence[str], Metric]], early_stopping_rounds: Optional[int], verbose: Union[int, bool], xgb_model: Optional[Union[XGBModel, Booster]], feature_weights: Optional[_DaskCollection], callbacks: Optional[Sequence[TrainingCallback]]) -> 'DaskXGBRanker':
        msg = 'Use `qid` instead of `group` on dask interface.'
        if not (group is None and eval_group is None):
            raise ValueError(msg)
        if qid is None:
            raise ValueError('`qid` is required for ranking.')
        params = self.get_xgb_params()
        dtrain, evals = await _async_wrap_evaluation_matrices(self.client, tree_method=self.tree_method, max_bin=self.max_bin, X=X, y=y, group=None, qid=qid, sample_weight=sample_weight, base_margin=base_margin, feature_weights=feature_weights, eval_set=eval_set, sample_weight_eval_set=sample_weight_eval_set, base_margin_eval_set=base_margin_eval_set, eval_group=None, eval_qid=eval_qid, missing=self.missing, enable_categorical=self.enable_categorical, feature_types=self.feature_types)
        if eval_metric is not None:
            if callable(eval_metric):
                raise ValueError('Custom evaluation metric is not yet supported for XGBRanker.')
        model, metric, params, early_stopping_rounds, callbacks = self._configure_fit(xgb_model, eval_metric, params, early_stopping_rounds, callbacks)
        results = await self.client.sync(_train_async, asynchronous=True, client=self.client, global_config=config.get_config(), dconfig=_get_dask_config(), params=params, dtrain=dtrain, num_boost_round=self.get_num_boosting_rounds(), evals=evals, obj=None, feval=None, custom_metric=metric, verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds, callbacks=callbacks, xgb_model=model)
        self._Booster = results['booster']
        self.evals_result_ = results['history']
        return self

    @_deprecate_positional_args
    def fit(self, X: _DataT, y: _DaskCollection, *, group: Optional[_DaskCollection]=None, qid: Optional[_DaskCollection]=None, sample_weight: Optional[_DaskCollection]=None, base_margin: Optional[_DaskCollection]=None, eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]]=None, eval_group: Optional[Sequence[_DaskCollection]]=None, eval_qid: Optional[Sequence[_DaskCollection]]=None, eval_metric: Optional[Union[str, Sequence[str], Callable]]=None, early_stopping_rounds: Optional[int]=None, verbose: Union[int, bool]=False, xgb_model: Optional[Union[XGBModel, Booster]]=None, sample_weight_eval_set: Optional[Sequence[_DaskCollection]]=None, base_margin_eval_set: Optional[Sequence[_DaskCollection]]=None, feature_weights: Optional[_DaskCollection]=None, callbacks: Optional[Sequence[TrainingCallback]]=None) -> 'DaskXGBRanker':
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        return self._client_sync(self._fit_async, **args)
    fit.__doc__ = XGBRanker.fit.__doc__