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
def _validate_and_convert_feature_col_as_array_col(dataset: DataFrame, features_col_name: str) -> Column:
    """It handles
    1. Convert vector type to array type
    2. Cast to Array(Float32)"""
    features_col_datatype = dataset.schema[features_col_name].dataType
    features_col = col(features_col_name)
    if isinstance(features_col_datatype, ArrayType):
        if not isinstance(features_col_datatype.elementType, (DoubleType, FloatType, LongType, IntegerType, ShortType)):
            raise ValueError('If feature column is array type, its elements must be number type.')
        features_array_col = features_col.cast(ArrayType(FloatType())).alias(alias.data)
    elif isinstance(features_col_datatype, VectorUDT):
        features_array_col = vector_to_array(features_col, dtype='float32').alias(alias.data)
    else:
        raise ValueError('feature column must be array type or `pyspark.ml.linalg.Vector` type, if you want to use multiple numetric columns as features, please use `pyspark.ml.transform.VectorAssembler` to assemble them into a vector type column first.')
    return features_array_col