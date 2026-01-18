import logging
import os
import pathlib
import re
import shutil
import tempfile
import urllib.parse
import zipfile
from io import BytesIO
from mlflow import tracking
from mlflow.entities import Param, SourceType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_RUN_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from mlflow.tracking import fluent
from mlflow.tracking.context.default_context import _get_user
from mlflow.utils.git_utils import get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
from mlflow.utils.rest_utils import augmented_raise_for_status
def _parse_file_uri(uri: str) -> str:
    """Converts file URIs to filesystem paths"""
    if _is_file_uri(uri):
        parsed_file_uri = urllib.parse.urlparse(uri)
        return str(pathlib.Path(parsed_file_uri.netloc, parsed_file_uri.path, parsed_file_uri.fragment))
    return uri