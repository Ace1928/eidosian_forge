import atexit
import contextlib
import importlib
import inspect
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import _get_store, artifact_utils
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.default_experiment import registry as default_experiment_registry
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.annotations import experimental
from mlflow.utils.async_logging.run_operations import RunOperations
from mlflow.utils.autologging_utils import (
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.import_hooks import register_post_import_hook
from mlflow.utils.mlflow_tags import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import _validate_experiment_id_type, _validate_run_id
def _set_experiment_primary_metric(experiment_id: str, primary_metric: str, greater_is_better: bool):
    client = MlflowClient()
    client.set_experiment_tag(experiment_id, MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME, primary_metric)
    client.set_experiment_tag(experiment_id, MLFLOW_EXPERIMENT_PRIMARY_METRIC_GREATER_IS_BETTER, str(greater_is_better))