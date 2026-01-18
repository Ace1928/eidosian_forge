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
def _create_pip_requirement(self, conda_env_path, pip_requirements_path):
    """
        This method creates a requirements.txt file for the model dependencies if the file does not
        already exist. It uses the pip dependencies found in the conda.yaml env file.

        Args:
            conda_env_path: Path to conda.yaml env file which contains the required pip
                dependencies
            pip_requirements_path: Path where the new requirements.txt will be created.
        """
    with open(conda_env_path) as f:
        conda_env = yaml.safe_load(f)
    pip_deps = _get_pip_deps(conda_env)
    _mlflow_additional_pip_env(pip_deps, pip_requirements_path)