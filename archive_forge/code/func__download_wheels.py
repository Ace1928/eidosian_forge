import os
import platform
import shutil
import subprocess
import sys
import yaml
import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.pyfunc.model import MLMODEL_FILE_NAME, Model
from mlflow.store.artifact.utils.models import _parse_model_uri, get_model_name_and_version
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
@classmethod
def _download_wheels(cls, pip_requirements_path, dst_path):
    """
        Downloads all the wheels of the dependencies specified in the requirements.txt file.
        The pip wheel download_command defaults to downloading only binary packages using
        the `--only-binary=:all:` option. This behavior can be overridden using an
        environment variable `MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS`, which will allows
        setting different options such as `--prefer-binary`, `--no-binary`, etc.

        Args:
            pip_requirements_path: Path to requirements.txt in the model directory
            dst_path: Path to the directory where the wheels are to be downloaded
        """
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    pip_wheel_options = MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS.get()
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', pip_wheel_options, '--wheel-dir', dst_path, '-r', pip_requirements_path, '--no-cache-dir'], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise MlflowException(f'An error occurred while downloading the dependency wheels: {e.stderr}')