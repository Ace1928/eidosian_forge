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
class LGBMRanker(LGBMModel):
    """LightGBM ranker.

    .. warning::

        scikit-learn doesn't support ranking applications yet,
        therefore this class is not really compatible with the sklearn ecosystem.
        Please use this class mainly for training and applying ranking models in common sklearnish way.
    """

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, group: Optional[_LGBM_GroupType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_group: Optional[List[_LGBM_GroupType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, eval_at: Union[List[int], Tuple[int, ...]]=(1, 2, 3, 4, 5), feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, LGBMModel]]=None) -> 'LGBMRanker':
        """Docstring is inherited from the LGBMModel."""
        if group is None:
            raise ValueError('Should set group for ranking task')
        if eval_set is not None:
            if eval_group is None:
                raise ValueError('Eval_group cannot be None when eval_set is not None')
            elif len(eval_group) != len(eval_set):
                raise ValueError('Length of eval_group should be equal to eval_set')
            elif isinstance(eval_group, dict) and any((i not in eval_group or eval_group[i] is None for i in range(len(eval_group)))) or (isinstance(eval_group, list) and any((group is None for group in eval_group))):
                raise ValueError('Should set group for all eval datasets for ranking task; if you use dict, the index should start from 0')
        self._eval_at = eval_at
        super().fit(X, y, sample_weight=sample_weight, init_score=init_score, group=group, eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_init_score=eval_init_score, eval_group=eval_group, eval_metric=eval_metric, feature_name=feature_name, categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self
    _base_doc = LGBMModel.fit.__doc__.replace('self : LGBMModel', 'self : LGBMRanker')
    fit.__doc__ = _base_doc[:_base_doc.find('eval_class_weight :')] + _base_doc[_base_doc.find('eval_init_score :'):]
    _base_doc = fit.__doc__
    _before_feature_name, _feature_name, _after_feature_name = _base_doc.partition('feature_name :')
    fit.__doc__ = f'{_before_feature_name}eval_at : list or tuple of int, optional (default=(1, 2, 3, 4, 5))\n        The evaluation positions of the specified metric.\n    {_feature_name}{_after_feature_name}'