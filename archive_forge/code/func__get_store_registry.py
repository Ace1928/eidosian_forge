from functools import partial
from mlflow.environment_variables import MLFLOW_REGISTRY_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.databricks_workspace_model_registry_rest_store import (
from mlflow.store.model_registry.file_store import FileStore
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service.utils import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import (
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
def _get_store_registry():
    global _model_registry_store_registry
    from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
    if _model_registry_store_registry is not None:
        return _model_registry_store_registry
    _model_registry_store_registry = ModelRegistryStoreRegistry()
    _model_registry_store_registry.register('databricks', _get_databricks_rest_store)
    _model_registry_store_registry.register(_DATABRICKS_UNITY_CATALOG_SCHEME, UcModelRegistryStore)
    for scheme in ['http', 'https']:
        _model_registry_store_registry.register(scheme, _get_rest_store)
    for scheme in DATABASE_ENGINES:
        _model_registry_store_registry.register(scheme, _get_sqlalchemy_store)
    for scheme in ['', 'file']:
        _model_registry_store_registry.register(scheme, _get_file_store)
    _model_registry_store_registry.register_entrypoints()
    return _model_registry_store_registry