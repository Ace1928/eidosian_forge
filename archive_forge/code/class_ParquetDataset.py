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
class ParquetDataset(_PandasConvertibleDataset):
    """
    Representation of a dataset in parquet format with files having the `.parquet` extension.
    """

    def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
        return read_parquet_as_pandas_df(data_parquet_path=local_data_file_path)

    @staticmethod
    def handles_format(dataset_format: str) -> bool:
        return dataset_format == 'parquet'