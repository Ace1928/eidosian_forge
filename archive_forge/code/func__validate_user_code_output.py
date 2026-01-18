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
def _validate_user_code_output(self, func, *args):
    import pandas as pd
    ingested_df = func(*args)
    if not isinstance(ingested_df, pd.DataFrame):
        raise MlflowException(message=f"The `ingested_data` is not a DataFrame, please make sure '{_USER_DEFINED_INGEST_STEP_MODULE}.{self.loader_method}' returns a Pandas DataFrame object.", error_code=INVALID_PARAMETER_VALUE) from None
    return ingested_df