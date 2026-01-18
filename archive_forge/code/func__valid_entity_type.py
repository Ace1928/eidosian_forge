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
def _valid_entity_type(cls, entity_type):
    entity_type = cls._trim_backticks(entity_type)
    if entity_type not in cls._VALID_IDENTIFIERS:
        raise MlflowException(f"Invalid entity type '{entity_type}'. Valid values are {cls._IDENTIFIERS}", error_code=INVALID_PARAMETER_VALUE)
    if entity_type in cls._ALTERNATE_PARAM_IDENTIFIERS:
        return cls._PARAM_IDENTIFIER
    elif entity_type in cls._ALTERNATE_METRIC_IDENTIFIERS:
        return cls._METRIC_IDENTIFIER
    elif entity_type in cls._ALTERNATE_TAG_IDENTIFIERS:
        return cls._TAG_IDENTIFIER
    elif entity_type in cls._ALTERNATE_ATTRIBUTE_IDENTIFIERS:
        return cls._ATTRIBUTE_IDENTIFIER
    elif entity_type in cls._ALTERNATE_DATASET_IDENTIFIERS:
        return cls._DATASET_IDENTIFIER
    else:
        return entity_type