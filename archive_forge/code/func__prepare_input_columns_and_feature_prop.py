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
def _prepare_input_columns_and_feature_prop(self, dataset: DataFrame) -> Tuple[List[Column], FeatureProp]:
    label_col = col(self.getOrDefault(self.labelCol)).alias(alias.label)
    select_cols = [label_col]
    features_cols_names = None
    enable_sparse_data_optim = self.getOrDefault(self.enable_sparse_data_optim)
    if enable_sparse_data_optim:
        features_col_name = self.getOrDefault(self.featuresCol)
        features_col_datatype = dataset.schema[features_col_name].dataType
        if not isinstance(features_col_datatype, VectorUDT):
            raise ValueError('If enable_sparse_data_optim is True, the feature column values must be `pyspark.ml.linalg.Vector` type.')
        select_cols.extend(_get_unwrapped_vec_cols(col(features_col_name)))
    elif self.getOrDefault(self.features_cols):
        features_cols_names = self.getOrDefault(self.features_cols)
        features_cols = _validate_and_convert_feature_col_as_float_col_list(dataset, features_cols_names)
        select_cols.extend(features_cols)
    else:
        features_array_col = _validate_and_convert_feature_col_as_array_col(dataset, self.getOrDefault(self.featuresCol))
        select_cols.append(features_array_col)
    if self.isDefined(self.weightCol) and self.getOrDefault(self.weightCol) != '':
        select_cols.append(col(self.getOrDefault(self.weightCol)).alias(alias.weight))
    has_validation_col = False
    if self.isDefined(self.validationIndicatorCol) and self.getOrDefault(self.validationIndicatorCol) != '':
        select_cols.append(col(self.getOrDefault(self.validationIndicatorCol)).alias(alias.valid))
        has_validation_col = True
    if self.isDefined(self.base_margin_col) and self.getOrDefault(self.base_margin_col) != '':
        select_cols.append(col(self.getOrDefault(self.base_margin_col)).alias(alias.margin))
    if self.isDefined(self.qid_col) and self.getOrDefault(self.qid_col) != '':
        select_cols.append(col(self.getOrDefault(self.qid_col)).alias(alias.qid))
    feature_prop = FeatureProp(enable_sparse_data_optim, has_validation_col, features_cols_names)
    return (select_cols, feature_prop)