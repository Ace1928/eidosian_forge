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
class _Dataset:
    """
    Base class representing an ingestable dataset.
    """

    def __init__(self, dataset_format: str):
        """
        Args:
            dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        """
        self.dataset_format = dataset_format

    @abstractmethod
    def resolve_to_parquet(self, dst_path: str):
        """
        Fetches the dataset, converts it to parquet, and stores it at the specified `dst_path`.

        Args:
            dst_path: The local filesystem path at which to store the resolved parquet dataset
                (e.g. `<execution_directory_path>/steps/ingest/outputs/dataset.parquet`).
        """
        pass

    @classmethod
    def from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> '_Dataset':
        """
        Constructs a dataset instance from the specified dataset configuration
        and recipe root path.

        Args:
            dataset_config: Dictionary representation of the recipe dataset configuration
                (i.e. the `data` section of recipe.yaml).
            recipe_root: The absolute path of the associated recipe root directory on the
                local filesystem.

        Returns:
            A `_Dataset` instance representing the configured dataset.
        """
        if not cls.handles_format(dataset_config.get('using')):
            raise MlflowException(f'Invalid format {dataset_config.get('using')} for dataset {cls}', error_code=INVALID_PARAMETER_VALUE)
        return cls._from_config(dataset_config, recipe_root)

    @classmethod
    @abstractmethod
    def _from_config(cls, dataset_config, recipe_root) -> '_Dataset':
        """
        Constructs a dataset instance from the specified dataset configuration
        and recipe root path.

        Args:
            dataset_config: Dictionary representation of the recipe dataset configuration
                (i.e. the `data` section of recipe.yaml).
            recipe_root: The absolute path of the associated recipe root directory on the
                local filesystem.

        Returns:
            A `_Dataset` instance representing the configured dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def handles_format(dataset_format: str) -> bool:
        """
        Determines whether or not the dataset class is a compatible representation of the
        specified dataset format.

        Args:
            dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).

        Returns:
            `True` if the dataset class is a compatible representation of the specified
            dataset format, `False` otherwise.
        """
        pass

    @classmethod
    def _get_required_config(cls, dataset_config: Dict[str, Any], key: str) -> Any:
        """
        Obtains the value associated with the specified dataset configuration key, first verifying
        that the key is present in the config and throwing if it is not.

        Args:
            dataset_config: Dictionary representation of the recipe dataset configuration
                (i.e. the `data` section of recipe.yaml).
            key: The key within the dataset configuration for which to fetch the associated
                value.

        Returns:
            The value associated with the specified configuration key.
        """
        try:
            return dataset_config[key]
        except KeyError:
            raise MlflowException(f"The `{key}` configuration key must be specified for dataset with using '{dataset_config.get('using')}' format") from None