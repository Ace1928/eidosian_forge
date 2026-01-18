import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType,
from .callback import _EvalResultDict, record_evaluation
from .compat import (SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
from .engine import train
class LGBMRegressor(_LGBMRegressorBase, LGBMModel):
    """LightGBM regressor."""

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, LGBMModel]]=None) -> 'LGBMRegressor':
        """Docstring is inherited from the LGBMModel."""
        super().fit(X, y, sample_weight=sample_weight, init_score=init_score, eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_init_score=eval_init_score, eval_metric=eval_metric, feature_name=feature_name, categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self
    _base_doc = LGBMModel.fit.__doc__.replace('self : LGBMModel', 'self : LGBMRegressor')
    _base_doc = _base_doc[:_base_doc.find('group :')] + _base_doc[_base_doc.find('eval_set :'):]
    _base_doc = _base_doc[:_base_doc.find('eval_class_weight :')] + _base_doc[_base_doc.find('eval_init_score :'):]
    fit.__doc__ = _base_doc[:_base_doc.find('eval_group :')] + _base_doc[_base_doc.find('eval_metric :'):]