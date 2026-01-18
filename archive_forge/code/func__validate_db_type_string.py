import logging
import numbers
import posixpath
import re
from typing import List
from mlflow.entities import Dataset, DatasetInput, InputTag, Param, RunTag
from mlflow.environment_variables import MLFLOW_TRUNCATE_LONG_VALUES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.string_utils import is_string_type
def _validate_db_type_string(db_type):
    """validates db_type parsed from DB URI is supported"""
    if db_type not in DATABASE_ENGINES:
        error_msg = f"Invalid database engine: '{db_type}'. '{_UNSUPPORTED_DB_TYPE_MSG}'"
        raise MlflowException(error_msg, INVALID_PARAMETER_VALUE)