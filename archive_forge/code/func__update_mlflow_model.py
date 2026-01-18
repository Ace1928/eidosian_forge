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
def _update_mlflow_model(self, original_model_metadata, mlflow_model):
    """
        Modifies the MLModel file to reflect updated information such as the run_id,
        utc_time_created. Additionally, this also adds `wheels` to the MLModel file to indicate that
        this is a `wheeled` model.

        Args:
            original_model_file_path: The model metadata stored in the original MLmodel file.
            mlflow_model: :py:mod:`mlflow.models.Model` configuration of the newly created
                          wheeled model
        """
    run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
    if mlflow_model is None:
        mlflow_model = Model(run_id=run_id)
    original_model_metadata.__dict__.update({k: v for k, v in mlflow_model.__dict__.items() if v})
    mlflow_model.__dict__.update(original_model_metadata.__dict__)
    mlflow_model.artifact_path = WheeledModel.get_wheel_artifact_path(mlflow_model.artifact_path)
    mlflow_model.wheels = {_PLATFORM: platform.platform()}
    return mlflow_model