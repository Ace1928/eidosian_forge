import functools
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar
import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
def _construct_databricks_model_version_url(host: str, name: str, version: str, workspace_id: Optional[str]=None) -> str:
    model_version_url = host
    if workspace_id and workspace_id != '0':
        model_version_url += '?o=' + str(workspace_id)
    model_version_url += f'#mlflow/models/{name}/versions/{version}'
    return model_version_url