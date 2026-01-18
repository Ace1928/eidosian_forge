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
class DatabricksRuntimeVersion(NamedTuple):
    is_client_image: bool
    major: int
    minor: int

    @classmethod
    def parse(cls):
        dbr_version = get_databricks_runtime_version()
        try:
            dbr_version_splits = dbr_version.split('.', maxsplit=2)
            if dbr_version_splits[0] == 'client':
                is_client_image = True
                major = int(dbr_version_splits[1])
                minor = int(dbr_version_splits[2]) if len(dbr_version_splits) > 2 else 0
            else:
                is_client_image = False
                major = int(dbr_version_splits[0])
                minor = int(dbr_version_splits[1])
            return cls(is_client_image, major, minor)
        except Exception:
            raise MlflowException(f"Failed to parse databricks runtime version '{dbr_version}'.")