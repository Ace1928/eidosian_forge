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
def _should_log_model(spark_model):
    from pyspark.ml.base import Model
    class_name = _get_fully_qualified_class_name(spark_model)
    should_log = class_name in _log_model_allowlist
    if not should_log:
        for name in _log_model_allowlist:
            if name.endswith('*') and class_name.startswith(name[:-1]):
                should_log = True
                break
    if should_log:
        if class_name == 'pyspark.ml.classification.OneVsRestModel':
            return _should_log_model(spark_model.models[0])
        elif class_name == 'pyspark.ml.pipeline.PipelineModel':
            return all((_should_log_model(stage) for stage in spark_model.stages if isinstance(stage, Model)))
        elif _is_parameter_search_model(spark_model):
            return _should_log_model(spark_model.bestModel)
        else:
            return all((_should_log_model(param_value) for _, param_value in _get_param_map(spark_model).items() if isinstance(param_value, Model)))
    else:
        return False