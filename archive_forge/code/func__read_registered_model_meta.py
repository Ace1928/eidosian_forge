import logging
import multiprocessing
import os
import shutil
import signal
import sys
from pathlib import Path
from subprocess import Popen, check_call
from typing import List
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DEPLOYMENT_FLAVOR_NAME, MLFLOW_DISABLE_ENV_CREATION
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc import _extract_conda_env, mlserver, scoring_server
from mlflow.store.artifact.models_artifact_repo import REGISTERED_MODEL_META_FILE_NAME
from mlflow.utils import env_manager as em
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.file_utils import read_yaml
from mlflow.utils.virtualenv import _get_or_create_virtualenv
from mlflow.version import VERSION as MLFLOW_VERSION
def _read_registered_model_meta(model_path):
    model_meta = {}
    if os.path.isfile(os.path.join(model_path, REGISTERED_MODEL_META_FILE_NAME)):
        model_meta = read_yaml(model_path, REGISTERED_MODEL_META_FILE_NAME)
    return model_meta