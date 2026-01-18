import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def _infer_model_signature(input_example_slice):
    input_slice_df = _find_and_set_features_col_as_vector_if_needed(spark.createDataFrame(input_example_slice), spark_model)
    model_output = spark_model.transform(input_slice_df).drop(*input_slice_df.columns)
    unsupported_columns = _get_columns_with_unsupported_data_type(model_output)
    if unsupported_columns:
        _logger.warning(f'Model outputs contain unsupported Spark data types: {unsupported_columns}. Output schema is not be logged.')
        model_output = None
    else:
        model_output = model_output.toPandas()
    return infer_signature(input_example_slice, model_output)