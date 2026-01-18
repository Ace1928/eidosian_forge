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
def _convert_to_parquet(self, dataset_file_paths: List[str], dst_path: str):
    import pandas as pd
    aggregated_dataframe = None
    for data_file_path in dataset_file_paths:
        _path = pathlib.Path(data_file_path)
        data_file_as_dataframe = self._load_file_as_pandas_dataframe(local_data_file_path=data_file_path)
        aggregated_dataframe = pd.concat([aggregated_dataframe, data_file_as_dataframe]) if aggregated_dataframe is not None else data_file_as_dataframe
    write_pandas_df_as_parquet(df=aggregated_dataframe, data_parquet_path=dst_path)