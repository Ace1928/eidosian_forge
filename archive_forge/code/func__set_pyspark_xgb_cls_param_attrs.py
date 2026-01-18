import warnings
from typing import Any, List, Optional, Type, Union
import numpy as np
from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from xgboost import XGBClassifier, XGBRanker, XGBRegressor
from .core import (  # type: ignore
from .utils import get_class_name
def _set_pyspark_xgb_cls_param_attrs(estimator: Type[_SparkXGBEstimator], model: Type[_SparkXGBModel]) -> None:
    """This function automatically infer to xgboost parameters and set them
    into corresponding pyspark estimators and models"""
    params_dict = estimator._get_xgb_params_default()

    def param_value_converter(v: Any) -> Any:
        if isinstance(v, np.generic):
            return np.array(v).item()
        if isinstance(v, dict):
            return {k: param_value_converter(nv) for k, nv in v.items()}
        if isinstance(v, list):
            return [param_value_converter(nv) for nv in v]
        return v

    def set_param_attrs(attr_name: str, param: Param) -> None:
        param.typeConverter = param_value_converter
        setattr(estimator, attr_name, param)
        setattr(model, attr_name, param)
    for name in params_dict.keys():
        doc = f'Refer to XGBoost doc of {get_class_name(estimator._xgb_cls())} for this param {name}'
        param_obj: Param = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)
    fit_params_dict = estimator._get_fit_params_default()
    for name in fit_params_dict.keys():
        doc = f'Refer to XGBoost doc of {get_class_name(estimator._xgb_cls())}.fit() for this param {name}'
        if name == 'callbacks':
            doc += 'The callbacks can be arbitrary functions. It is saved using cloudpickle which is not a fully self-contained format. It may fail to load with different versions of dependencies.'
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)
    predict_params_dict = estimator._get_predict_params_default()
    for name in predict_params_dict.keys():
        doc = f'Refer to XGBoost doc of {get_class_name(estimator._xgb_cls())}.predict() for this param {name}'
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)