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
def _serve_mleap():
    serve_cmd = ['java', '-cp', '"/opt/java/jars/*"', 'org.mlflow.sagemaker.ScoringServer', MODEL_PATH, str(DEFAULT_SAGEMAKER_SERVER_PORT)]
    serve_cmd = ' '.join(serve_cmd)
    mleap = Popen(serve_cmd, shell=True)
    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[mleap.pid]))
    awaited_pids = _await_subprocess_exit_any(procs=[mleap])
    _sigterm_handler(awaited_pids)