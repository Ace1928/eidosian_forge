import contextlib
import json
import logging
import os
import re
import sys
import warnings
from datetime import timedelta
import click
from click import UsageError
import mlflow.db
import mlflow.deployments.cli
import mlflow.experiments
import mlflow.runs
import mlflow.store.artifact.cli
from mlflow import projects, version
from mlflow.entities import ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME
from mlflow.exceptions import InvalidUrlException, MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.tracking import _get_store
from mlflow.utils import cli_args
from mlflow.utils.logging_utils import eprint
from mlflow.utils.os import get_entry_points, is_windows
from mlflow.utils.process import ShellCommandException
from mlflow.utils.server_cli_utils import (
def _validate_static_prefix(ctx, param, value):
    """
    Validate that the static_prefix option starts with a "/" and does not end in a "/".
    Conforms to the callback interface of click documented at
    http://click.pocoo.org/5/options/#callbacks-for-validation.
    """
    if value is not None:
        if not value.startswith('/'):
            raise UsageError("--static-prefix must begin with a '/'.")
        if value.endswith('/'):
            raise UsageError("--static-prefix should not end with a '/'.")
    return value