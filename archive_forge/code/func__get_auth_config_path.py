import configparser
from pathlib import Path
from typing import NamedTuple
from mlflow.environment_variables import MLFLOW_AUTH_CONFIG_PATH
def _get_auth_config_path() -> str:
    return MLFLOW_AUTH_CONFIG_PATH.get() or Path(__file__).parent.joinpath('basic_auth.ini').resolve()