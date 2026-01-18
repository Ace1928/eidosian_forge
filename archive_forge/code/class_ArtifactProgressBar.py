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
class ArtifactProgressBar:

    def __init__(self, desc, total, step, **kwargs) -> None:
        self.desc = desc
        self.total = total
        self.step = step
        self.pbar = None
        self.progress = 0
        self.kwargs = kwargs

    def set_pbar(self):
        if MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR.get():
            try:
                from tqdm.auto import tqdm
                self.pbar = tqdm(total=self.total, desc=self.desc, **self.kwargs)
            except ImportError:
                pass

    @classmethod
    def chunks(cls, file_size, desc, chunk_size):
        bar = cls(desc, total=file_size, step=chunk_size, unit='iB', unit_scale=True, unit_divisor=1024, miniters=1)
        if file_size >= _PROGRESS_BAR_DISPLAY_THRESHOLD:
            bar.set_pbar()
        return bar

    @classmethod
    def files(cls, desc, total):
        bar = cls(desc, total=total, step=1)
        bar.set_pbar()
        return bar

    def update(self):
        if self.pbar:
            update_step = min(self.total - self.progress, self.step)
            self.pbar.update(update_step)
            self.pbar.refresh()
            self.progress += update_step

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.pbar:
            self.pbar.close()