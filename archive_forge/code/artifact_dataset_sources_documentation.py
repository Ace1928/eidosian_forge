import re
import warnings
from pathlib import Path
from typing import Any, Dict, TypeVar
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_registered_artifact_repositories
from mlflow.utils.uri import is_local_uri

            Returns:
                A JSON-compatible dictionary representation of the {dataset_source_name}.
            