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
def _create_child_runs_for_parameter_search(parent_estimator, parent_model, parent_run, child_tags):
    client = MlflowClient()
    child_run_start_time = parent_run.info.start_time
    child_run_end_time = get_current_time_millis()
    estimator_param_maps = parent_estimator.getEstimatorParamMaps()
    tuned_estimator = parent_estimator.getEstimator()
    metrics_dict, _ = _get_param_search_metrics_and_best_index(parent_estimator, parent_model)
    for i, est_param in enumerate(estimator_param_maps):
        child_estimator = tuned_estimator.copy(est_param)
        tags_to_log = dict(child_tags) if child_tags else {}
        tags_to_log.update({MLFLOW_PARENT_RUN_ID: parent_run.info.run_id})
        tags_to_log.update(_get_estimator_info_tags(child_estimator))
        child_run = client.create_run(experiment_id=parent_run.info.experiment_id, start_time=child_run_start_time, tags=tags_to_log)
        params_to_log = _get_instance_param_map(child_estimator, parent_estimator._autologging_metadata.uid_to_indexed_name_map)
        param_batches_to_log = _chunk_dict(params_to_log, chunk_size=MAX_PARAMS_TAGS_PER_BATCH)
        metrics_to_log = {k: v[i] for k, v in metrics_dict.items()}
        for params_batch, metrics_batch in zip_longest(param_batches_to_log, [metrics_to_log], fillvalue={}):
            truncated_params_batch = _truncate_dict(params_batch, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
            truncated_metrics_batch = _truncate_dict(metrics_batch, max_key_length=MAX_ENTITY_KEY_LENGTH)
            client.log_batch(run_id=child_run.info.run_id, params=[Param(str(key), str(value)) for key, value in truncated_params_batch.items()], metrics=[Metric(key=str(key), value=value, timestamp=child_run_end_time, step=0) for key, value in truncated_metrics_batch.items()])
        client.set_terminated(run_id=child_run.info.run_id, end_time=child_run_end_time)