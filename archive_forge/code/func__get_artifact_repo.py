import os
from collections import OrderedDict
from itertools import zip_longest
from typing import List, Optional
from mlflow.entities import ExperimentTag, Metric, Param, RunStatus, RunTag, ViewType
from mlflow.entities.dataset_input import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking import GET_METRIC_HISTORY_MAX_RESULTS, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking._tracking_service import utils
from mlflow.tracking.metric_value_conversion_utils import convert_metric_value_to_float_if_possible
from mlflow.utils import chunk_list
from mlflow.utils.async_logging.run_operations import RunOperations, get_combined_run_operations
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri
from mlflow.utils.validation import (
def _get_artifact_repo(self, run_id):
    cached_repo = TrackingServiceClient._artifact_repos_cache.get(run_id)
    if cached_repo is not None:
        return cached_repo
    else:
        run = self.get_run(run_id)
        artifact_uri = add_databricks_profile_info_to_artifact_uri(run.info.artifact_uri, self.tracking_uri)
        artifact_repo = get_artifact_repository(artifact_uri)
        if len(TrackingServiceClient._artifact_repos_cache) > 1024:
            TrackingServiceClient._artifact_repos_cache.popitem(last=False)
        TrackingServiceClient._artifact_repos_cache[run_id] = artifact_repo
        return artifact_repo