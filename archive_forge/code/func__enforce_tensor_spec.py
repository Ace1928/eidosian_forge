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
def _enforce_tensor_spec(values: Union[np.ndarray, 'csc_matrix', 'csr_matrix'], tensor_spec: TensorSpec):
    """
    Enforce the input tensor shape and type matches the provided tensor spec.
    """
    expected_shape = tensor_spec.shape
    expected_type = tensor_spec.type
    actual_shape = values.shape
    actual_type = values.dtype if isinstance(values, np.ndarray) else values.data.dtype
    if len(expected_shape) == 1 and expected_shape[0] == -1 and (expected_type == np.dtype('O')):
        return values
    if len(expected_shape) > 1 and -1 in expected_shape[1:] and (len(actual_shape) == 1) and (actual_type == np.dtype('O')):
        return values
    if len(expected_shape) != len(actual_shape):
        raise MlflowException(f'Shape of input {actual_shape} does not match expected shape {expected_shape}.')
    for expected, actual in zip(expected_shape, actual_shape):
        if expected == -1:
            continue
        if expected != actual:
            raise MlflowException(f'Shape of input {actual_shape} does not match expected shape {expected_shape}.')
    if clean_tensor_type(actual_type) != expected_type:
        raise MlflowException(f'dtype of input {actual_type} does not match expected dtype {expected_type}')
    return values