import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
class MlflowCreds(NamedTuple):
    username: Optional[str]
    password: Optional[str]