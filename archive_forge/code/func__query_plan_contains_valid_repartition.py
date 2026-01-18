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
def _query_plan_contains_valid_repartition(self, dataset: DataFrame) -> bool:
    """
        Returns true if the latest element in the logical plan is a valid repartition
        The logic plan string format is like:

        == Optimized Logical Plan ==
        Repartition 4, true
        +- LogicalRDD [features#12, label#13L], false

        i.e., the top line in the logical plan is the last operation to execute.
        so, in this method, we check the first line, if it is a "Repartition" operation,
        and the result dataframe has the same partition number with num_workers param,
        then it means the dataframe is well repartitioned and we don't need to
        repartition the dataframe again.
        """
    num_partitions = dataset.rdd.getNumPartitions()
    assert dataset._sc._jvm is not None
    query_plan = dataset._sc._jvm.PythonSQLUtils.explainString(dataset._jdf.queryExecution(), 'extended')
    start = query_plan.index('== Optimized Logical Plan ==')
    start += len('== Optimized Logical Plan ==') + 1
    num_workers = self.getOrDefault(self.num_workers)
    if query_plan[start:start + len('Repartition')] == 'Repartition' and num_workers == num_partitions:
        return True
    return False