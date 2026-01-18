import json
import logging
import os
import yaml
import mlflow.projects.databricks
import mlflow.utils.uri
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.backend import loader
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import (
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.mlflow_tags import (
def _validate_execution_environment(project, backend):
    if project.docker_env and backend == 'databricks':
        raise ExecutionException('Running docker-based projects on Databricks is not yet supported.')