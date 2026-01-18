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
def _validate_input_tag(input_tag: InputTag):
    if input_tag is None:
        raise MlflowException('InputTag cannot be None', INVALID_PARAMETER_VALUE)
    if input_tag.key is None:
        raise MlflowException('InputTag key cannot be None', INVALID_PARAMETER_VALUE)
    if input_tag.value is None:
        raise MlflowException('InputTag value cannot be None', INVALID_PARAMETER_VALUE)
    if len(input_tag.key) > MAX_INPUT_TAG_KEY_SIZE:
        raise MlflowException(f'InputTag key exceeds the maximum length of {MAX_INPUT_TAG_KEY_SIZE}', INVALID_PARAMETER_VALUE)
    if len(input_tag.value) > MAX_INPUT_TAG_VALUE_SIZE:
        raise MlflowException(f'InputTag value exceeds the maximum length of {MAX_INPUT_TAG_VALUE_SIZE}', INVALID_PARAMETER_VALUE)