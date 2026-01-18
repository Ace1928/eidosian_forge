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
def _validate_registered_model_tag(key, value):
    """
    Check that a tag with the specified key & value is valid and raise an exception if it isn't.
    """
    _validate_tag_name(key)
    _validate_length_limit('Registered model key', MAX_MODEL_REGISTRY_TAG_KEY_LENGTH, key)
    _validate_length_limit('Registered model value', MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH, value)