import json
import posixpath
from typing import Any, Dict, Iterator, Optional
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils import AttrDict
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
class DatabricksEndpoint(AttrDict):
    """
    A dictionary-like object representing a Databricks serving endpoint.

    .. code-block:: python

        endpoint = DatabricksEndpoint(
            {
                "name": "chat",
                "creator": "alice@company.com",
                "creation_timestamp": 0,
                "last_updated_timestamp": 0,
                "state": {...},
                "config": {...},
                "tags": [...],
                "id": "88fd3f75a0d24b0380ddc40484d7a31b",
            }
        )
        assert endpoint.name == "chat"
    """
    pass