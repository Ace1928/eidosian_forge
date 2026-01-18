from typing import Any, List
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import insecure_hash
Computes a normalized digest for a list of hashable elements.

    Args:
        elements: A list of hashable elements for inclusion in the md5 digest.

    Returns:
        An 8-character, truncated md5 digest.
    