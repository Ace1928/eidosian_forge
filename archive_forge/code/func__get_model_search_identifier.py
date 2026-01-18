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
def _get_model_search_identifier(cls, identifier, valid_attributes):
    tokens = identifier.split('.', maxsplit=1)
    if len(tokens) == 1:
        key = tokens[0]
        identifier = cls._ATTRIBUTE_IDENTIFIER
    else:
        entity_type, key = tokens
        valid_entity_types = ('attribute', 'tag', 'tags')
        if entity_type not in valid_entity_types:
            raise MlflowException.invalid_parameter_value(f"Invalid entity type '{entity_type}'. Valid entity types are {valid_entity_types}")
        identifier = cls._TAG_IDENTIFIER if entity_type in ('tag', 'tags') else cls._ATTRIBUTE_IDENTIFIER
    if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
        raise MlflowException.invalid_parameter_value(f"Invalid attribute key '{key}' specified. Valid keys are '{valid_attributes}'")
    key = cls._trim_backticks(cls._strip_quotes(key))
    return {'type': identifier, 'key': key}