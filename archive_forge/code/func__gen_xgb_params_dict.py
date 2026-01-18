import base64
import json
import logging
import os
from collections import namedtuple
from typing import (
import numpy as np
import pandas as pd
from pyspark import RDD, SparkContext, cloudpickle
from pyspark.ml import Estimator, Model
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
from pyspark.ml.util import (
from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, countDistinct, pandas_udf, rand, struct
from pyspark.sql.types import (
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module
import xgboost
from xgboost import XGBClassifier
from xgboost.compat import is_cudf_available, is_cupy_available
from xgboost.core import Booster, _check_distributed_params
from xgboost.sklearn import DEFAULT_N_ESTIMATORS, XGBModel, _can_use_qdm
from xgboost.training import train as worker_train
from .._typing import ArrayLike
from .data import (
from .params import (
from .utils import (
def _gen_xgb_params_dict(self, gen_xgb_sklearn_estimator_param: bool=False) -> Dict[str, Any]:
    """Generate the xgboost parameters which will be passed into xgboost library"""
    xgb_params = {}
    non_xgb_params = set(_pyspark_specific_params) | self._get_fit_params_default().keys() | self._get_predict_params_default().keys()
    if not gen_xgb_sklearn_estimator_param:
        non_xgb_params |= set(_non_booster_params)
    for param in self.extractParamMap():
        if param.name not in non_xgb_params:
            xgb_params[param.name] = self.getOrDefault(param)
    arbitrary_params_dict = self.getOrDefault(self.getParam('arbitrary_params_dict'))
    xgb_params.update(arbitrary_params_dict)
    return xgb_params