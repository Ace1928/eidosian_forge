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
class _LocationBasedDataset(_Dataset):
    """
    Base class representing an ingestable dataset with a configurable `location` attribute.
    """

    def __init__(self, location: Union[str, List[str]], dataset_format: str, recipe_root: str):
        """
        Args:
            location: The location of the dataset (one dataset as a string or list of multiple
                datasets)
                (e.g. '/tmp/myfile.parquet', './mypath', 's3://mybucket/mypath', or YAML list:
                    location:
                        - http://www.myserver.com/dataset/df1.csv
                        - http://www.myserver.com/dataset/df1.csv
                )

            dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).

            recipe_root: The absolute path of the associated recipe root directory on the local
                filesystem.
        """
        super().__init__(dataset_format=dataset_format)
        self.location = _LocationBasedDataset._sanitize_local_dataset_multiple_locations_if_necessary(dataset_location=location, recipe_root=recipe_root)

    @abstractmethod
    def resolve_to_parquet(self, dst_path: str):
        pass

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> '_Dataset':
        return cls(location=cls._get_required_config(dataset_config=dataset_config, key='location'), recipe_root=recipe_root, dataset_format=cls._get_required_config(dataset_config=dataset_config, key='using'))

    @staticmethod
    def _sanitize_local_dataset_multiple_locations_if_necessary(dataset_location: Union[str, List[str]], recipe_root: str) -> List[str]:
        if isinstance(dataset_location, str):
            return [_LocationBasedDataset._sanitize_local_dataset_location_if_necessary(dataset_location, recipe_root)]
        elif isinstance(dataset_location, list):
            return [_LocationBasedDataset._sanitize_local_dataset_location_if_necessary(locaton, recipe_root) for locaton in dataset_location]
        else:
            raise MlflowException(f'Unsupported location type: {type(dataset_location)}')

    @staticmethod
    def _sanitize_local_dataset_location_if_necessary(dataset_location: str, recipe_root: str) -> str:
        """
        Checks whether or not the specified `dataset_location` is a local filesystem location and,
        if it is, converts it to an absolute path if it is not already absolute.

        Args:
            dataset_location: The dataset location from the recipe dataset configuration.
            recipe_root: The absolute path of the recipe root directory on the local
                filesystem.

        Returns:
            The sanitized dataset location.
        """
        local_dataset_path_or_none = get_local_path_or_none(path_or_uri=dataset_location)
        if local_dataset_path_or_none is None:
            return dataset_location
        local_dataset_path = local_file_uri_to_path(uri=local_dataset_path_or_none)
        local_dataset_path = pathlib.Path(local_dataset_path)
        if local_dataset_path.is_absolute():
            return str(local_dataset_path)
        else:
            return str(pathlib.Path(recipe_root) / local_dataset_path)

    @staticmethod
    @abstractmethod
    def handles_format(dataset_format: str) -> bool:
        pass