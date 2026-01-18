import ast
import base64
import json
import math
import operator
import re
import shlex
import sqlparse
from packaging.version import Version
from sqlparse.sql import (
from sqlparse.tokens import Token as TokenType
from mlflow.entities import RunInfo
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.utils.mlflow_tags import (
@classmethod
def _get_value_for_sort(cls, run, key_type, key, ascending):
    """Returns a tuple suitable to be used as a sort key for runs."""
    sort_value = None
    key = SearchUtils.translate_key_alias(key)
    if key_type == cls._METRIC_IDENTIFIER:
        sort_value = run.data.metrics.get(key)
    elif key_type == cls._PARAM_IDENTIFIER:
        sort_value = run.data.params.get(key)
    elif key_type == cls._TAG_IDENTIFIER:
        sort_value = run.data.tags.get(key)
    elif key_type == cls._ATTRIBUTE_IDENTIFIER:
        sort_value = getattr(run.info, key)
    else:
        raise MlflowException(f"Invalid order_by entity type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
    is_none = sort_value is None
    is_nan = isinstance(sort_value, float) and math.isnan(sort_value)
    fill_value = (1 if ascending else -1) * math.inf
    if is_none:
        sort_value = fill_value
    elif is_nan:
        sort_value = -fill_value
    is_none_or_nan = is_none or is_nan
    return (is_none_or_nan, sort_value) if ascending else (not is_none_or_nan, sort_value)