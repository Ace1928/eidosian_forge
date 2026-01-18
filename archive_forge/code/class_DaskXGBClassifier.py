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
@xgboost_model_doc('Implementation of the scikit-learn API for XGBoost classification.', ['estimators', 'model'])
class DaskXGBClassifier(DaskScikitLearnBase, XGBClassifierBase):

    async def _fit_async(self, X: _DataT, y: _DaskCollection, sample_weight: Optional[_DaskCollection], base_margin: Optional[_DaskCollection], eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]], eval_metric: Optional[Union[str, Sequence[str], Metric]], sample_weight_eval_set: Optional[Sequence[_DaskCollection]], base_margin_eval_set: Optional[Sequence[_DaskCollection]], early_stopping_rounds: Optional[int], verbose: Union[int, bool], xgb_model: Optional[Union[Booster, XGBModel]], feature_weights: Optional[_DaskCollection], callbacks: Optional[Sequence[TrainingCallback]]) -> 'DaskXGBClassifier':
        params = self.get_xgb_params()
        dtrain, evals = await _async_wrap_evaluation_matrices(self.client, tree_method=self.tree_method, max_bin=self.max_bin, X=X, y=y, group=None, qid=None, sample_weight=sample_weight, base_margin=base_margin, feature_weights=feature_weights, eval_set=eval_set, sample_weight_eval_set=sample_weight_eval_set, base_margin_eval_set=base_margin_eval_set, eval_group=None, eval_qid=None, missing=self.missing, enable_categorical=self.enable_categorical, feature_types=self.feature_types)
        if isinstance(y, da.Array):
            self.classes_ = await self.client.compute(da.unique(y))
        else:
            self.classes_ = await self.client.compute(y.drop_duplicates())
        if _is_cudf_ser(self.classes_):
            self.classes_ = self.classes_.to_cupy()
        if _is_cupy_array(self.classes_):
            self.classes_ = self.classes_.get()
        self.classes_ = numpy.array(self.classes_)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = self.n_classes_
        else:
            params['objective'] = 'binary:logistic'
        if callable(self.objective):
            obj: Optional[Callable] = _objective_decorator(self.objective)
        else:
            obj = None
        model, metric, params, early_stopping_rounds, callbacks = self._configure_fit(xgb_model, eval_metric, params, early_stopping_rounds, callbacks)
        results = await self.client.sync(_train_async, asynchronous=True, client=self.client, global_config=config.get_config(), dconfig=_get_dask_config(), params=params, dtrain=dtrain, num_boost_round=self.get_num_boosting_rounds(), evals=evals, obj=obj, feval=None, custom_metric=metric, verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds, callbacks=callbacks, xgb_model=model)
        self._Booster = results['booster']
        if not callable(self.objective):
            self.objective = params['objective']
        self._set_evaluation_result(results['history'])
        return self

    def fit(self, X: _DataT, y: _DaskCollection, *, sample_weight: Optional[_DaskCollection]=None, base_margin: Optional[_DaskCollection]=None, eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]]=None, eval_metric: Optional[Union[str, Sequence[str], Callable]]=None, early_stopping_rounds: Optional[int]=None, verbose: Union[int, bool]=True, xgb_model: Optional[Union[Booster, XGBModel]]=None, sample_weight_eval_set: Optional[Sequence[_DaskCollection]]=None, base_margin_eval_set: Optional[Sequence[_DaskCollection]]=None, feature_weights: Optional[_DaskCollection]=None, callbacks: Optional[Sequence[TrainingCallback]]=None) -> 'DaskXGBClassifier':
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        return self._client_sync(self._fit_async, **args)

    async def _predict_proba_async(self, X: _DataT, validate_features: bool, base_margin: Optional[_DaskCollection], iteration_range: Optional[Tuple[int, int]]) -> _DaskCollection:
        if self.objective == 'multi:softmax':
            raise ValueError("multi:softmax doesn't support `predict_proba`.  Switch to `multi:softproba` instead")
        predts = await super()._predict_async(data=X, output_margin=False, validate_features=validate_features, base_margin=base_margin, iteration_range=iteration_range)
        vstack = update_wrapper(partial(da.vstack, allow_unknown_chunksizes=True), da.vstack)
        return _cls_predict_proba(getattr(self, 'n_classes_', 0), predts, vstack)

    def predict_proba(self, X: _DaskCollection, validate_features: bool=True, base_margin: Optional[_DaskCollection]=None, iteration_range: Optional[Tuple[int, int]]=None) -> Any:
        _assert_dask_support()
        return self._client_sync(self._predict_proba_async, X=X, validate_features=validate_features, base_margin=base_margin, iteration_range=iteration_range)
    predict_proba.__doc__ = XGBClassifier.predict_proba.__doc__

    async def _predict_async(self, data: _DataT, output_margin: bool, validate_features: bool, base_margin: Optional[_DaskCollection], iteration_range: Optional[Tuple[int, int]]) -> _DaskCollection:
        pred_probs = await super()._predict_async(data, output_margin, validate_features, base_margin, iteration_range)
        if output_margin:
            return pred_probs
        if len(pred_probs.shape) == 1:
            preds = (pred_probs > 0.5).astype(int)
        else:
            assert len(pred_probs.shape) == 2
            assert isinstance(pred_probs, da.Array)

            def _argmax(x: Any) -> Any:
                return x.argmax(axis=1)
            preds = da.map_blocks(_argmax, pred_probs, drop_axis=1)
        return preds