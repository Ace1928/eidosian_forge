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
def _get_experiment_id_from_env():
    experiment_name = MLFLOW_EXPERIMENT_NAME.get()
    experiment_id = MLFLOW_EXPERIMENT_ID.get()
    if experiment_name is not None:
        exp = MlflowClient().get_experiment_by_name(experiment_name)
        if exp:
            if experiment_id and experiment_id != exp.experiment_id:
                raise MlflowException(message=f'The provided {MLFLOW_EXPERIMENT_ID} environment variable value `{experiment_id}` does not match the experiment id `{exp.experiment_id}` for experiment name `{experiment_name}`', error_code=INVALID_PARAMETER_VALUE)
            else:
                return exp.experiment_id
        else:
            return MlflowClient().create_experiment(name=experiment_name)
    if experiment_id is not None:
        try:
            exp = MlflowClient().get_experiment(experiment_id)
            return exp.experiment_id
        except MlflowException as exc:
            raise MlflowException(message=f'The provided {MLFLOW_EXPERIMENT_ID} environment variable value `{experiment_id}` does not exist in the tracking server. Provide a valid experiment_id.', error_code=INVALID_PARAMETER_VALUE) from exc