import click
from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import is_local_uri
def _is_default_backend_store_uri(backend_store_uri: str) -> bool:
    """Utility function to validate if the configured backend store uri location is set as the
    default value for MLflow server.

    Args:
        backend_store_uri: The value set for the backend store uri for MLflow server artifact
            handling.

    Returns:
        bool True if the default value is set.

    """
    return backend_store_uri == DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH