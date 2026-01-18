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
def _reshape_and_cast_pandas_column_values(name, pd_series, tensor_spec):
    if tensor_spec.shape[0] != -1 or -1 in tensor_spec.shape[1:]:
        raise MlflowException(f'For pandas dataframe input, the first dimension of shape must be a variable dimension and other dimensions must be fixed, but in model signature the shape of {('input ' + name if name else 'the unnamed input')} is {tensor_spec.shape}.')
    if np.isscalar(pd_series[0]):
        for shape in [(-1,), (-1, 1)]:
            if tensor_spec.shape == shape:
                return _enforce_tensor_spec(np.array(pd_series, dtype=tensor_spec.type).reshape(shape), tensor_spec)
        raise MlflowException(f"The input pandas dataframe column '{name}' contains scalar values, which requires the shape to be (-1,) or (-1, 1), but got tensor spec shape of {tensor_spec.shape}.", error_code=INVALID_PARAMETER_VALUE)
    elif isinstance(pd_series[0], list) and np.isscalar(pd_series[0][0]):
        reshape_err_msg = f"The value in the Input DataFrame column '{name}' could not be converted to the expected shape of: '{tensor_spec.shape}'. Ensure that each of the input list elements are of uniform length and that the data can be coerced to the tensor type '{tensor_spec.type}'"
        try:
            flattened_numpy_arr = np.vstack(pd_series.tolist())
            reshaped_numpy_arr = flattened_numpy_arr.reshape(tensor_spec.shape).astype(tensor_spec.type)
        except ValueError:
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        if len(reshaped_numpy_arr) != len(pd_series):
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        return reshaped_numpy_arr
    elif isinstance(pd_series[0], np.ndarray):
        reshape_err_msg = f"The value in the Input DataFrame column '{name}' could not be converted to the expected shape of: '{tensor_spec.shape}'. Ensure that each of the input numpy array elements are of uniform length and can be reshaped to above expected shape."
        try:
            reshaped_numpy_arr = np.vstack(pd_series.tolist()).reshape(tensor_spec.shape)
        except ValueError:
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        if len(reshaped_numpy_arr) != len(pd_series):
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        return reshaped_numpy_arr
    else:
        raise MlflowException('Because the model signature requires tensor spec input, the input pandas dataframe values should be either scalar value, python list containing scalar values or numpy array containing scalar values, other types are not supported.', error_code=INVALID_PARAMETER_VALUE)