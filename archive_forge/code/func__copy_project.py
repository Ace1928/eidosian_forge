import atexit
import codecs
import errno
import fnmatch
import gzip
import json
import logging
import math
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import CalledProcessError, TimeoutExpired
from typing import Optional, Union
from urllib.parse import unquote
from urllib.request import pathname2url
import yaml
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MissingConfigException, MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialType
from mlflow.utils import download_cloud_file_chunk, merge_dicts
from mlflow.utils.databricks_utils import _get_dbutils
from mlflow.utils.os import is_windows
from mlflow.utils.process import cache_return_value_per_process
from mlflow.utils.request_utils import cloud_storage_http_request, download_chunk
from mlflow.utils.rest_utils import augmented_raise_for_status
def _copy_project(src_path, dst_path=''):
    """Internal function used to copy MLflow project during development.

    Copies the content of the whole directory tree except patterns defined in .dockerignore.
    The MLflow is assumed to be accessible as a local directory in this case.

    Args:
        dst_path: MLflow will be copied here

    Returns:
        Name of the MLflow project directory.
    """

    def _docker_ignore(mlflow_root):
        docker_ignore = os.path.join(mlflow_root, '.dockerignore')
        patterns = []
        if os.path.exists(docker_ignore):
            with open(docker_ignore) as f:
                patterns = [x.strip() for x in f.readlines()]

        def ignore(_, names):
            res = set()
            for p in patterns:
                res.update(set(fnmatch.filter(names, p)))
            return list(res)
        return ignore if patterns else None
    mlflow_dir = 'mlflow-project'
    assert os.path.isfile(os.path.join(src_path, 'pyproject.toml')), 'file not found ' + str(os.path.abspath(os.path.join(src_path, 'pyproject.toml')))
    shutil.copytree(src_path, os.path.join(dst_path, mlflow_dir), ignore=_docker_ignore(src_path))
    return mlflow_dir