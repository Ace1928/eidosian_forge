import os
from subprocess import PIPE, STDOUT, Popen
from typing import Optional, Union
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DOCKER_OPENJDK_VERSION
from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.version import VERSION
def _pip_mlflow_install_step(dockerfile_context_dir, mlflow_home):
    """
    Get docker build commands for installing MLflow given a Docker context dir and optional source
    directory
    """
    if mlflow_home:
        mlflow_dir = _copy_project(src_path=os.path.abspath(mlflow_home), dst_path=dockerfile_context_dir)
        return f'# Install MLflow from local source\nCOPY {mlflow_dir} /opt/mlflow\nRUN pip install /opt/mlflow'
    else:
        return f'# Install MLflow\nRUN pip install mlflow=={VERSION}'