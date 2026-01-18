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
def _cast_schema_type(input_data, schema=None):
    input_data = deepcopy(input_data)
    types_dict = schema.input_dict() if schema and schema.has_input_names() else {}
    if schema is not None:
        if len(types_dict) == 1 and isinstance(input_data, list) and (not any((isinstance(x, dict) for x in input_data))):
            input_data = {next(iter(types_dict)): input_data}
        elif not schema.has_input_names() and (not isinstance(input_data, list)):
            raise MlflowInvalidInputException(f'Failed to parse input data. This model contains an un-named tensor-based model signature which expects a single n-dimensional array as input, however, an input of type {type(input_data)} was found.')
    if isinstance(input_data, dict):
        input_data = {col: convert_data_type(data, types_dict.get(col)) for col, data in input_data.items()}
    elif isinstance(input_data, list):
        if all((isinstance(x, dict) for x in input_data)):
            input_data = [{col: convert_data_type(value, types_dict.get(col)) for col, value in data.items()} for data in input_data]
        else:
            spec = schema.inputs[0] if schema else None
            input_data = convert_data_type(input_data, spec)
    else:
        spec = schema.inputs[0] if schema else None
        try:
            input_data = convert_data_type(input_data, spec)
        except Exception as e:
            raise MlflowInvalidInputException(f'Failed to parse input data. This model contains a tensor-based model signature with input names, which suggests a dictionary / a list of dictionaries input mapping input name to tensor or a pure list, but an input of `{input_data}` was found.') from e
    return input_data