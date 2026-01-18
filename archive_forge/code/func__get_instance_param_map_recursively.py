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
def _get_instance_param_map_recursively(instance, level, uid_to_indexed_name_map):
    from pyspark.ml.param import Params
    from pyspark.ml.pipeline import Pipeline
    param_map = _get_param_map(instance)
    expanded_param_map = {}
    is_pipeline = isinstance(instance, Pipeline)
    is_parameter_search_estimator = _is_parameter_search_estimator(instance)
    logged_param_name_prefix = '' if level == 0 else uid_to_indexed_name_map[instance.uid] + '.'
    for param_name, param_value in param_map.items():
        logged_param_name = logged_param_name_prefix + param_name
        if is_pipeline and param_name == 'stages':
            expanded_param_map[logged_param_name] = [uid_to_indexed_name_map[stage.uid] for stage in instance.getStages()]
            for stage in instance.getStages():
                stage_param_map = _get_instance_param_map_recursively(stage, level + 1, uid_to_indexed_name_map)
                expanded_param_map.update(stage_param_map)
        elif is_parameter_search_estimator and param_name == 'estimator':
            expanded_param_map[logged_param_name] = uid_to_indexed_name_map[param_value.uid]
        elif is_parameter_search_estimator and param_name == 'estimatorParamMaps':
            pass
        elif isinstance(param_value, Params):
            expanded_param_map[logged_param_name] = uid_to_indexed_name_map[param_value.uid]
            internal_param_map = _get_instance_param_map_recursively(param_value, level + 1, uid_to_indexed_name_map)
            expanded_param_map.update(internal_param_map)
        else:
            expanded_param_map[logged_param_name] = param_value
    return expanded_param_map