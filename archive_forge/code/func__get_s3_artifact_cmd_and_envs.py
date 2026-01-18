import logging
import os
import platform
import posixpath
import subprocess
import sys
from pathlib import Path
import mlflow
from mlflow import tracking
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.projects import env_type
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import LocalSubmittedRun
from mlflow.projects.utils import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_or_create_conda_env
from mlflow.utils.databricks_utils import get_databricks_env_vars, is_in_databricks_runtime
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.file_utils import get_or_create_nfs_tmp_dir
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENV
from mlflow.utils.os import is_windows
from mlflow.utils.virtualenv import (
def _get_s3_artifact_cmd_and_envs(artifact_repo):
    if platform.system() == 'Windows':
        win_user_dir = os.environ['USERPROFILE']
        aws_path = os.path.join(win_user_dir, '.aws')
    else:
        aws_path = posixpath.expanduser('~/.aws')
    volumes = []
    if posixpath.exists(aws_path):
        volumes = ['-v', '{}:{}'.format(str(aws_path), '/.aws')]
    envs = {'AWS_SECRET_ACCESS_KEY': os.environ.get('AWS_SECRET_ACCESS_KEY'), 'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID'), 'MLFLOW_S3_ENDPOINT_URL': os.environ.get('MLFLOW_S3_ENDPOINT_URL'), 'MLFLOW_S3_IGNORE_TLS': os.environ.get('MLFLOW_S3_IGNORE_TLS')}
    envs = {k: v for k, v in envs.items() if v is not None}
    return (volumes, envs)