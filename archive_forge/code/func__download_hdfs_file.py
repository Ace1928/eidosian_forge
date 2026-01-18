import os
import posixpath
import tempfile
import urllib.parse
from contextlib import contextmanager
import packaging.version
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import mkdir, relative_path_to_artifact_path
def _download_hdfs_file(hdfs, remote_file_path, local_file_path):
    dirs = os.path.dirname(local_file_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    with open(local_file_path, 'wb') as f:
        f.write(hdfs.open(remote_file_path, 'rb').read())