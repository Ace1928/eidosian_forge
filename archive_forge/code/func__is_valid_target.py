import urllib
from typing import Optional
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path
def _is_valid_target(target: str):
    """
    Evaluates the basic structure of a provided target to determine if the scheme and
    netloc are provided
    """
    if target == 'databricks':
        return True
    return _is_valid_uri(target)