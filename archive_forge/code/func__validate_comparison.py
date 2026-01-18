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
def _validate_comparison(cls, tokens):
    base_error_string = 'Invalid comparison clause'
    if len(tokens) != 3:
        raise MlflowException(f'{base_error_string}. Expected 3 tokens found {len(tokens)}', error_code=INVALID_PARAMETER_VALUE)
    if not isinstance(tokens[0], Identifier):
        raise MlflowException(f"{base_error_string}. Expected 'Identifier' found '{tokens[0]}'", error_code=INVALID_PARAMETER_VALUE)
    if not isinstance(tokens[1], Token) and tokens[1].ttype != TokenType.Operator.Comparison:
        raise MlflowException(f"{base_error_string}. Expected comparison found '{tokens[1]}'", error_code=INVALID_PARAMETER_VALUE)
    if not isinstance(tokens[2], Token) and (tokens[2].ttype not in cls.STRING_VALUE_TYPES.union(cls.NUMERIC_VALUE_TYPES) or isinstance(tokens[2], Identifier)):
        raise MlflowException(f"{base_error_string}. Expected value token found '{tokens[2]}'", error_code=INVALID_PARAMETER_VALUE)