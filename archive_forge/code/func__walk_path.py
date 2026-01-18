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
def _walk_path(self, hdfs, hdfs_path):
    if hdfs.exists(hdfs_path):
        if hdfs.isdir(hdfs_path):
            for subdir, _, files in hdfs.walk(hdfs_path):
                if subdir != hdfs_path:
                    yield (subdir, hdfs.isdir(subdir), hdfs.info(subdir).get('size'))
                for f in files:
                    file_path = posixpath.join(subdir, f)
                    yield (file_path, hdfs.isdir(file_path), hdfs.info(file_path).get('size'))
        else:
            yield (hdfs_path, False, hdfs.info(hdfs_path).get('size'))