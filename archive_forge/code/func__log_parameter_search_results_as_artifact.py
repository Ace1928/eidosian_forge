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
def _log_parameter_search_results_as_artifact(param_maps, metrics_dict, run_id):
    import pandas as pd
    result_dict = defaultdict(list)
    result_dict['params'] = []
    result_dict.update(metrics_dict)
    for param_map in param_maps:
        result_dict['params'].append(json.dumps(param_map))
        for param_name, param_value in param_map.items():
            result_dict[f'param.{param_name}'].append(param_value)
    results_df = pd.DataFrame.from_dict(result_dict)
    with TempDir() as t:
        results_path = t.path('search_results.csv')
        results_df.to_csv(results_path, index=False)
        MlflowClient().log_artifact(run_id, results_path)