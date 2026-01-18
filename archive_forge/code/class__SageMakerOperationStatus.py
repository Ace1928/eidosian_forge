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
class _SageMakerOperationStatus:
    STATE_SUCCEEDED = 'succeeded'
    STATE_FAILED = 'failed'
    STATE_IN_PROGRESS = 'in progress'
    STATE_TIMED_OUT = 'timed_out'

    def __init__(self, state, message):
        self.state = state
        self.message = message

    @classmethod
    def in_progress(cls, message=None):
        if message is None:
            message = 'The operation is still in progress.'
        return cls(_SageMakerOperationStatus.STATE_IN_PROGRESS, message)

    @classmethod
    def timed_out(cls, duration_seconds):
        return cls(_SageMakerOperationStatus.STATE_TIMED_OUT, f'Timed out after waiting {duration_seconds} seconds for the operation to complete. This operation may still be in progress. Please check the AWS console for more information.')

    @classmethod
    def failed(cls, message):
        return cls(_SageMakerOperationStatus.STATE_FAILED, message)

    @classmethod
    def succeeded(cls, message):
        return cls(_SageMakerOperationStatus.STATE_SUCCEEDED, message)