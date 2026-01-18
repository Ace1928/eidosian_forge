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
def _does_registered_model_match_clauses(cls, model, sed):
    key_type = sed.get('type')
    key = sed.get('key')
    value = sed.get('value')
    comparator = sed.get('comparator').upper()
    if cls.is_string_attribute(key_type, key, comparator):
        lhs = getattr(model, key)
    elif cls.is_numeric_attribute(key_type, key, comparator):
        lhs = getattr(model, key)
        value = int(value)
    elif cls.is_tag(key_type, comparator):
        lhs = model.tags.get(key, None)
    else:
        raise MlflowException(f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
    if lhs is None:
        return False
    return SearchUtils.get_comparison_func(comparator)(lhs, value)