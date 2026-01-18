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
class SparkXGBModelReader(MLReader):
    """
    Spark Xgboost model reader.
    """

    def __init__(self, cls: Type['_SparkXGBModel']) -> None:
        super().__init__()
        self.cls = cls
        self.logger = get_logger(self.__class__.__name__, level='WARN')

    def load(self, path: str) -> '_SparkXGBModel':
        """
        Load metadata and model for a :py:class:`_SparkXGBModel`

        :return: SparkXGBRegressorModel or SparkXGBClassifierModel instance
        """
        _, py_model = _SparkXGBSharedReadWrite.loadMetadataAndInstance(self.cls, path, self.sc, self.logger)
        py_model = cast('_SparkXGBModel', py_model)
        xgb_sklearn_params = py_model._gen_xgb_params_dict(gen_xgb_sklearn_estimator_param=True)
        model_load_path = os.path.join(path, 'model')
        ser_xgb_model = _get_spark_session().sparkContext.textFile(model_load_path).collect()[0]

        def create_xgb_model() -> 'XGBModel':
            return self.cls._xgb_cls()(**xgb_sklearn_params)
        xgb_model = deserialize_xgb_model(ser_xgb_model, create_xgb_model)
        py_model._xgb_sklearn_model = xgb_model
        return py_model