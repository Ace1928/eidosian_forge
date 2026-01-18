import operator
import socket
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse
import numpy as np
import scipy.sparse as ss
from .basic import LightGBMError, _choose_param_value, _ConfigAliases, _log_info, _log_warning
from .compat import (DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED, Client, Future, LGBMNotFittedError, concat,
from .sklearn import (LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor, _LGBM_ScikitCustomObjectiveFunction,
class DaskLGBMRegressor(LGBMRegressor, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMRegressor."""

    def __init__(self, boosting_type: str='gbdt', num_leaves: int=31, max_depth: int=-1, learning_rate: float=0.1, n_estimators: int=100, subsample_for_bin: int=200000, objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]]=None, class_weight: Optional[Union[dict, str]]=None, min_split_gain: float=0.0, min_child_weight: float=0.001, min_child_samples: int=20, subsample: float=1.0, subsample_freq: int=0, colsample_bytree: float=1.0, reg_alpha: float=0.0, reg_lambda: float=0.0, random_state: Optional[Union[int, np.random.RandomState, 'np.random.Generator']]=None, n_jobs: Optional[int]=None, importance_type: str='split', client: Optional[Client]=None, **kwargs: Any):
        """Docstring is inherited from the lightgbm.LGBMRegressor.__init__."""
        self.client = client
        super().__init__(boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, subsample_for_bin=subsample_for_bin, objective=objective, class_weight=class_weight, min_split_gain=min_split_gain, min_child_weight=min_child_weight, min_child_samples=min_child_samples, subsample=subsample, subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=random_state, n_jobs=n_jobs, importance_type=importance_type, **kwargs)
    _base_doc = LGBMRegressor.__init__.__doc__
    _before_kwargs, _kwargs, _after_kwargs = _base_doc.partition('**kwargs')
    __init__.__doc__ = f'\n        {_before_kwargs}client : dask.distributed.Client or None, optional (default=None)\n        {' ':4}Dask client. If ``None``, ``distributed.default_client()`` will be used at runtime. The Dask client used by this class will not be saved if the model object is pickled.\n        {_kwargs}{_after_kwargs}\n        '

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_dask_getstate()

    def fit(self, X: _DaskMatrixLike, y: _DaskCollection, sample_weight: Optional[_DaskVectorLike]=None, init_score: Optional[_DaskVectorLike]=None, eval_set: Optional[List[Tuple[_DaskMatrixLike, _DaskCollection]]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_DaskVectorLike]]=None, eval_init_score: Optional[List[_DaskVectorLike]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, **kwargs: Any) -> 'DaskLGBMRegressor':
        """Docstring is inherited from the lightgbm.LGBMRegressor.fit."""
        self._lgb_dask_fit(model_factory=LGBMRegressor, X=X, y=y, sample_weight=sample_weight, init_score=init_score, eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_init_score=eval_init_score, eval_metric=eval_metric, **kwargs)
        return self
    _base_doc = _lgbmmodel_doc_fit.format(X_shape='Dask Array or Dask DataFrame of shape = [n_samples, n_features]', y_shape='Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]', sample_weight_shape='Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)', init_score_shape='Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)', group_shape='Dask Array or Dask Series or None, optional (default=None)', eval_sample_weight_shape='list of Dask Array or Dask Series, or None, optional (default=None)', eval_init_score_shape='list of Dask Array or Dask Series, or None, optional (default=None)', eval_group_shape='list of Dask Array or Dask Series, or None, optional (default=None)')
    _base_doc = _base_doc[:_base_doc.find('group :')] + _base_doc[_base_doc.find('eval_set :'):]
    _base_doc = _base_doc[:_base_doc.find('eval_class_weight :')] + _base_doc[_base_doc.find('eval_init_score :'):]
    _base_doc = _base_doc[:_base_doc.find('eval_group :')] + _base_doc[_base_doc.find('eval_metric :'):]
    fit.__doc__ = f'{_base_doc[:_base_doc.find('callbacks :')]}**kwargs\n        Other parameters passed through to ``LGBMRegressor.fit()``.\n\n    Returns\n    -------\n    self : lightgbm.DaskLGBMRegressor\n        Returns self.\n\n    {_lgbmmodel_doc_custom_eval_note}\n        '

    def predict(self, X: _DaskMatrixLike, raw_score: bool=False, start_iteration: int=0, num_iteration: Optional[int]=None, pred_leaf: bool=False, pred_contrib: bool=False, validate_features: bool=False, **kwargs: Any) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMRegressor.predict."""
        return _predict(model=self.to_local(), data=X, client=_get_dask_client(self.client), raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration, pred_leaf=pred_leaf, pred_contrib=pred_contrib, validate_features=validate_features, **kwargs)
    predict.__doc__ = _lgbmmodel_doc_predict.format(description='Return the predicted value for each sample.', X_shape='Dask Array or Dask DataFrame of shape = [n_samples, n_features]', output_name='predicted_result', predicted_result_shape='Dask Array of shape = [n_samples]', X_leaves_shape='Dask Array of shape = [n_samples, n_trees]', X_SHAP_values_shape='Dask Array of shape = [n_samples, n_features + 1]')

    def to_local(self) -> LGBMRegressor:
        """Create regular version of lightgbm.LGBMRegressor from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRegressor
            Local underlying model.
        """
        return self._lgb_dask_to_local(LGBMRegressor)