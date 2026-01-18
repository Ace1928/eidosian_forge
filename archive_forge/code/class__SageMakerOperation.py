import json
import logging
import os
import platform
import signal
import sys
import tarfile
import time
import urllib.parse
from subprocess import Popen
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.container import (
from mlflow.models.container import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import get_unique_resource_id
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data
class _SageMakerOperation:

    def __init__(self, status_check_fn, cleanup_fn):
        self.status_check_fn = status_check_fn
        self.cleanup_fn = cleanup_fn
        self.start_time = time.time()
        self.status = _SageMakerOperationStatus(_SageMakerOperationStatus.STATE_IN_PROGRESS, None)
        self.cleaned_up = False

    def await_completion(self, timeout_seconds):
        iteration = 0
        begin = time.time()
        while time.time() - begin < timeout_seconds:
            status = self.status_check_fn()
            if status.state == _SageMakerOperationStatus.STATE_IN_PROGRESS:
                if iteration % 4 == 0:
                    _logger.info(status.message)
                time.sleep(5)
                iteration += 1
                continue
            else:
                self.status = status
                return status
        duration_seconds = time.time() - begin
        return _SageMakerOperationStatus.timed_out(duration_seconds)

    def clean_up(self):
        if self.status.state != _SageMakerOperationStatus.STATE_SUCCEEDED:
            raise ValueError(f'Cannot clean up an operation that has not succeeded! Current operation state: {self.status.state}')
        if not self.cleaned_up:
            self.cleaned_up = True
        else:
            raise ValueError('`clean_up()` has already been executed for this operation!')
        self.cleanup_fn()