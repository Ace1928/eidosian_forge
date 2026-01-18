import datetime as dt
import decimal
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.types.utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import (
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
def _enforce_object(data: Dict[str, Any], obj: Object, required=True):
    if not required and data is None:
        return None
    if HAS_PYSPARK and isinstance(data, Row):
        data = data.asDict(True)
    if not isinstance(data, dict):
        raise MlflowException(f"Failed to enforce schema of '{data}' with type '{obj}'. Expected data to be dictionary, got {type(data).__name__}")
    if not isinstance(obj, Object):
        raise MlflowException(f"Failed to enforce schema of '{data}' with type '{obj}'. Expected obj to be Object, got {type(obj).__name__}")
    properties = {prop.name: prop for prop in obj.properties}
    required_props = {k for k, prop in properties.items() if prop.required}
    missing_props = required_props - set(data.keys())
    if missing_props:
        raise MlflowException(f'Missing required properties: {missing_props}')
    if (invalid_props := (data.keys() - properties.keys())):
        raise MlflowException(f'Invalid properties not defined in the schema found: {invalid_props}')
    for k, v in data.items():
        try:
            data[k] = _enforce_property(v, properties[k])
        except MlflowException as e:
            raise MlflowException(f'Failed to enforce schema for key `{k}`. Expected type {properties[k].to_dict()[k]['type']}, received type {type(v).__name__}') from e
    return data