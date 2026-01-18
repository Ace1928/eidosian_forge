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
def _validate_batch_log_api_req(json_req):
    if len(json_req) > MAX_BATCH_LOG_REQUEST_SIZE:
        error_msg = 'Batched logging API requests must be at most {limit} bytes, got a request of size {size}.'.format(limit=MAX_BATCH_LOG_REQUEST_SIZE, size=len(json_req))
        raise MlflowException(error_msg, error_code=INVALID_PARAMETER_VALUE)