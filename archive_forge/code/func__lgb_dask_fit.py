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
def _lgb_dask_fit(self, model_factory: Type[LGBMModel], X: _DaskMatrixLike, y: _DaskCollection, sample_weight: Optional[_DaskVectorLike]=None, init_score: Optional[_DaskCollection]=None, group: Optional[_DaskVectorLike]=None, eval_set: Optional[List[Tuple[_DaskMatrixLike, _DaskCollection]]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_DaskVectorLike]]=None, eval_class_weight: Optional[List[Union[dict, str]]]=None, eval_init_score: Optional[List[_DaskCollection]]=None, eval_group: Optional[List[_DaskVectorLike]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, eval_at: Optional[Union[List[int], Tuple[int, ...]]]=None, **kwargs: Any) -> '_DaskLGBMModel':
    if not DASK_INSTALLED:
        raise LightGBMError('dask is required for lightgbm.dask')
    if not all((DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED)):
        raise LightGBMError('dask, pandas and scikit-learn are required for lightgbm.dask')
    params = self.get_params(True)
    params.pop('client', None)
    model = _train(client=_get_dask_client(self.client), data=X, label=y, params=params, model_factory=model_factory, sample_weight=sample_weight, init_score=init_score, group=group, eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_class_weight=eval_class_weight, eval_init_score=eval_init_score, eval_group=eval_group, eval_metric=eval_metric, eval_at=eval_at, **kwargs)
    self.set_params(**model.get_params())
    self._lgb_dask_copy_extra_params(model, self)
    return self