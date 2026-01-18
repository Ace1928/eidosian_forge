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
def _gen_stage_hierarchy_recursively(stage, uid_to_indexed_name_map):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import OneVsRest
    stage_name = uid_to_indexed_name_map[stage.uid]
    if isinstance(stage, Pipeline):
        sub_stages = []
        for sub_stage in stage.getStages():
            sub_hierarchy = _gen_stage_hierarchy_recursively(sub_stage, uid_to_indexed_name_map)
            sub_stages.append(sub_hierarchy)
        return {'name': stage_name, 'stages': sub_stages}
    elif isinstance(stage, OneVsRest):
        classifier_hierarchy = _gen_stage_hierarchy_recursively(stage.getClassifier(), uid_to_indexed_name_map)
        return {'name': stage_name, 'classifier': classifier_hierarchy}
    elif _is_parameter_search_estimator(stage):
        evaluator = stage.getEvaluator()
        tuned_estimator = stage.getEstimator()
        return {'name': stage_name, 'evaluator': _gen_stage_hierarchy_recursively(evaluator, uid_to_indexed_name_map), 'tuned_estimator': _gen_stage_hierarchy_recursively(tuned_estimator, uid_to_indexed_name_map)}
    elif any(_get_stage_type_params(stage)):
        sub_params = {}
        for param_name, param_value in _get_stage_type_params(stage).items():
            sub_hierarchy = _gen_stage_hierarchy_recursively(param_value, uid_to_indexed_name_map)
            sub_params[param_name] = sub_hierarchy
        return {'name': stage_name, 'params': sub_params}
    else:
        return {'name': stage_name}