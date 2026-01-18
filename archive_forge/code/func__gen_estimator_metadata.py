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
def _gen_estimator_metadata(estimator):
    """
    Returns an `_AutologgingEstimatorMetadata` object.
    The `_AutologgingEstimatorMetadata` object includes:
     - hierarchy: the hierarchy of the estimator, it will expand
         pipeline/meta estimator/param tuning estimator
     - uid_to_indexed_name_map: a map of `uid` -> `name`, each nested instance uid will be
         mapped to a fixed name. The naming rule is using `{class_name}` if the
         instance type occurs once, or `{class_name}_{index}` if the instance type occurs
         multiple times. The index order is in line with depth-first traversing.
     - param_search_estimators: a list includes all param search estimators in the
         hierarchy tree.
    """
    uid_to_indexed_name_map = _get_uid_to_indexed_name_map(estimator)
    param_search_estimators = [stage for stage in _traverse_stage(estimator) if _is_parameter_search_estimator(stage)]
    hierarchy = _gen_stage_hierarchy_recursively(estimator, uid_to_indexed_name_map)
    metadata = _AutologgingEstimatorMetadata(hierarchy=hierarchy, uid_to_indexed_name_map=uid_to_indexed_name_map, param_search_estimators=param_search_estimators)
    estimator._autologging_metadata = metadata
    return metadata