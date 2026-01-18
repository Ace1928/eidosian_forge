import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
def _read_mlflow_creds_from_file() -> Tuple[Optional[str], Optional[str]]:
    path = _get_credentials_path()
    if not os.path.exists(path):
        return (None, None)
    config = configparser.ConfigParser()
    config.read(path)
    if 'mlflow' not in config:
        return (None, None)
    mlflow_cfg = config['mlflow']
    username_key = MLFLOW_TRACKING_USERNAME.name.lower()
    password_key = MLFLOW_TRACKING_PASSWORD.name.lower()
    return (mlflow_cfg.get(username_key), mlflow_cfg.get(password_key))