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
def _enforce_map(data: Any, map_type: Map, required=True):
    if not required and data is None:
        return None
    if not isinstance(data, dict):
        raise MlflowException(f'Expected data to be a dict, got {type(data).__name__}')
    if not all((isinstance(k, str) for k in data)):
        raise MlflowException('Expected all keys in the map type data are string type.')
    return {k: _enforce_type(v, map_type.value_type) for k, v in data.items()}