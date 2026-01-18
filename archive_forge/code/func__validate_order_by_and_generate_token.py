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
def _validate_order_by_and_generate_token(cls, order_by):
    try:
        parsed = sqlparse.parse(order_by)
    except Exception:
        raise MlflowException(f"Error on parsing order_by clause '{order_by}'", error_code=INVALID_PARAMETER_VALUE)
    if len(parsed) != 1 or not isinstance(parsed[0], Statement):
        raise MlflowException(f"Invalid order_by clause '{order_by}'. Could not be parsed.", error_code=INVALID_PARAMETER_VALUE)
    statement = parsed[0]
    ttype_for_timestamp = TokenType.Name.Builtin if Version(sqlparse.__version__) >= Version('0.4.3') else TokenType.Keyword
    if len(statement.tokens) == 1 and isinstance(statement[0], Identifier):
        token_value = statement.tokens[0].value
    elif len(statement.tokens) == 1 and statement.tokens[0].match(ttype=ttype_for_timestamp, values=[cls.ORDER_BY_KEY_TIMESTAMP]):
        token_value = cls.ORDER_BY_KEY_TIMESTAMP
    elif statement.tokens[0].match(ttype=ttype_for_timestamp, values=[cls.ORDER_BY_KEY_TIMESTAMP]) and all((token.is_whitespace for token in statement.tokens[1:-1])) and (statement.tokens[-1].ttype == TokenType.Keyword.Order):
        token_value = cls.ORDER_BY_KEY_TIMESTAMP + ' ' + statement.tokens[-1].value
    else:
        raise MlflowException(f"Invalid order_by clause '{order_by}'. Could not be parsed.", error_code=INVALID_PARAMETER_VALUE)
    return token_value