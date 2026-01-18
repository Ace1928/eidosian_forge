import logging
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Union
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
@contextmanager
def _use_tracking_uri(uri: str) -> None:
    """Temporarily use the specified tracking URI.

    Args:
        uri: The tracking URI to use.

    """
    global _tracking_uri
    old_tracking_uri = _tracking_uri
    try:
        _tracking_uri = uri
        yield
    finally:
        _tracking_uri = old_tracking_uri