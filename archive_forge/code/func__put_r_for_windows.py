import os
import posixpath
import sys
import threading
import urllib.parse
from contextlib import contextmanager
from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository
def _put_r_for_windows(sftp, local_dir, remote_dir, preserve_mtime=False):
    for entry in os.listdir(local_dir):
        local_path = os.path.join(local_dir, entry)
        remote_path = posixpath.join(remote_dir, entry)
        if os.path.isdir(local_path):
            sftp.mkdir(remote_path)
            _put_r_for_windows(sftp, local_path, remote_path, preserve_mtime)
        else:
            sftp.put(local_path, remote_path, preserve_mtime=preserve_mtime)