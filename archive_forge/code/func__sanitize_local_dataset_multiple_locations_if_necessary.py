import importlib
import logging
import os
import pathlib
import posixpath
import sys
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import (
from mlflow.utils._spark_utils import (
from mlflow.utils.file_utils import (
@staticmethod
def _sanitize_local_dataset_multiple_locations_if_necessary(dataset_location: Union[str, List[str]], recipe_root: str) -> List[str]:
    if isinstance(dataset_location, str):
        return [_LocationBasedDataset._sanitize_local_dataset_location_if_necessary(dataset_location, recipe_root)]
    elif isinstance(dataset_location, list):
        return [_LocationBasedDataset._sanitize_local_dataset_location_if_necessary(locaton, recipe_root) for locaton in dataset_location]
    else:
        raise MlflowException(f'Unsupported location type: {type(dataset_location)}')