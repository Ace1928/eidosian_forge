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
def _parse_list_from_sql_token(cls, token):
    try:
        str_or_tuple = ast.literal_eval(token.value)
        return [str_or_tuple] if isinstance(str_or_tuple, str) else str_or_tuple
    except SyntaxError:
        raise MlflowException('While parsing a list in the query, expected a non-empty list of string values, but got ill-formed list.', error_code=INVALID_PARAMETER_VALUE)