import warnings
from typing import Any, List, Optional
import entrypoints
from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
Parses and returns a DatasetSource object from its JSON representation.

        Args:
            source_json: The JSON representation of the DatasetSource.
            source_type: The string type of the DatasetSource, which indicates how to parse the
                source JSON.
        