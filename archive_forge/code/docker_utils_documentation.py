import os
from subprocess import PIPE, STDOUT, Popen
from typing import Optional, Union
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DOCKER_OPENJDK_VERSION
from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.version import VERSION

    Get docker build commands for installing MLflow given a Docker context dir and optional source
    directory
    