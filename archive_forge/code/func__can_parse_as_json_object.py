import base64
import json
import requests
from mlflow.environment_variables import (
from mlflow.exceptions import (
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.request_utils import (
from mlflow.utils.string_utils import strip_suffix
def _can_parse_as_json_object(string):
    try:
        return isinstance(json.loads(string), dict)
    except Exception:
        return False