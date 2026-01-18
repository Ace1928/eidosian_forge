import base64
import datetime
import importlib
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONEncoder
from typing import Any, Dict, Optional
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToJson, ParseDict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
def _merge_json_dicts(from_dict, to_dict):
    """Merges the json elements of from_dict into to_dict. Only works for json dicts
    converted from proto messages
    """
    for key, value in from_dict.items():
        if isinstance(key, int) and str(key) in to_dict:
            to_dict[key] = to_dict[str(key)]
            del to_dict[str(key)]
        if key not in to_dict:
            continue
        if isinstance(value, dict):
            _merge_json_dicts(from_dict[key], to_dict[key])
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, dict):
                    _merge_json_dicts(v, to_dict[key][i])
                else:
                    to_dict[key][i] = v
        else:
            to_dict[key] = from_dict[key]
    return to_dict