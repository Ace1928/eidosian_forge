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
def _get_databricks_uc_rest_store(store_uri, **_):
    from mlflow.exceptions import MlflowException
    from mlflow.version import VERSION
    global _tracking_store_registry
    supported_schemes = [scheme for scheme in _tracking_store_registry._registry if scheme != _DATABRICKS_UNITY_CATALOG_SCHEME]
    raise MlflowException(f"Detected Unity Catalog tracking URI '{store_uri}'. Setting the tracking URI to a Unity Catalog backend is not supported in the current version of the MLflow client ({VERSION}). Please specify a different tracking URI via mlflow.set_tracking_uri, with one of the supported schemes: {supported_schemes}. If you're trying to access models in the Unity Catalog, please upgrade to the latest version of the MLflow Python client, then specify a Unity Catalog model registry URI via mlflow.set_registry_uri('{_DATABRICKS_UNITY_CATALOG_SCHEME}') or mlflow.set_registry_uri('{_DATABRICKS_UNITY_CATALOG_SCHEME}://profile_name'), where 'profile_name' is the name of the Databricks CLI profile to use for authentication. Be sure to leave the tracking URI configured to use one of the supported schemes listed above.")