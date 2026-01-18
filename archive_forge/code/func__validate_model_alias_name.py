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
def _validate_model_alias_name(model_alias_name):
    if model_alias_name is None or model_alias_name == '':
        raise MlflowException('Registered model alias name cannot be empty.', INVALID_PARAMETER_VALUE)
    if not _REGISTERED_MODEL_ALIAS_REGEX.match(model_alias_name):
        raise MlflowException(f"Invalid alias name: '{model_alias_name}'. {_BAD_ALIAS_CHARACTERS_MESSAGE}", INVALID_PARAMETER_VALUE)
    _validate_length_limit('Registered model alias name', MAX_REGISTERED_MODEL_ALIAS_LENGTH, model_alias_name)
    if model_alias_name.lower() == 'latest':
        raise MlflowException("'latest' alias name (case insensitive) is reserved.", INVALID_PARAMETER_VALUE)
    if _REGISTERED_MODEL_ALIAS_VERSION_REGEX.match(model_alias_name):
        raise MlflowException(f"Version alias name '{model_alias_name}' is reserved.", INVALID_PARAMETER_VALUE)