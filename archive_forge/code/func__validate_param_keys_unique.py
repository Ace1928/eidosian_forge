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
def _validate_param_keys_unique(params):
    """Ensures that duplicate param keys are not present in the `log_batch()` params argument"""
    unique_keys = []
    dupe_keys = []
    for param in params:
        if param.key not in unique_keys:
            unique_keys.append(param.key)
        else:
            dupe_keys.append(param.key)
    if dupe_keys:
        raise MlflowException(f'Duplicate parameter keys have been submitted: {dupe_keys}. Please ensure the request contains only one param value per param key.', INVALID_PARAMETER_VALUE)