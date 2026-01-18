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
def _enforce_tensor_schema(pf_input: PyFuncInput, input_schema: Schema):
    """Enforce the input tensor(s) conforms to the model's tensor-based signature."""

    def _is_sparse_matrix(x):
        if not HAS_SCIPY:
            return False
        return isinstance(x, (csr_matrix, csc_matrix))
    if input_schema.has_input_names():
        if isinstance(pf_input, dict):
            new_pf_input = {}
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                if not isinstance(pf_input[col_name], np.ndarray):
                    raise MlflowException(f'This model contains a tensor-based model signature with input names, which suggests a dictionary input mapping input name to a numpy array, but a dict with value type {type(pf_input[col_name])} was found.', error_code=INVALID_PARAMETER_VALUE)
                new_pf_input[col_name] = _enforce_tensor_spec(pf_input[col_name], tensor_spec)
        elif isinstance(pf_input, pd.DataFrame):
            new_pf_input = {}
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                pd_series = pf_input[col_name]
                new_pf_input[col_name] = _reshape_and_cast_pandas_column_values(col_name, pd_series, tensor_spec)
        else:
            raise MlflowException(f'This model contains a tensor-based model signature with input names, which suggests a dictionary input mapping input name to tensor, or a pandas DataFrame input containing columns mapping input name to flattened list value from tensor, but an input of type {type(pf_input)} was found.', error_code=INVALID_PARAMETER_VALUE)
    else:
        tensor_spec = input_schema.inputs[0]
        if isinstance(pf_input, pd.DataFrame):
            num_input_columns = len(pf_input.columns)
            if pf_input.empty:
                raise MlflowException('Input DataFrame is empty.')
            elif num_input_columns == 1:
                new_pf_input = _reshape_and_cast_pandas_column_values(None, pf_input[pf_input.columns[0]], tensor_spec)
            else:
                if tensor_spec.shape != (-1, num_input_columns):
                    raise MlflowException(f'This model contains a model signature with an unnamed input. Since the input data is a pandas DataFrame containing multiple columns, the input shape must be of the structure (-1, number_of_dataframe_columns). Instead, the input DataFrame passed had {num_input_columns} columns and an input shape of {tensor_spec.shape} with all values within the DataFrame of scalar type. Please adjust the passed in DataFrame to match the expected structure', error_code=INVALID_PARAMETER_VALUE)
                new_pf_input = _enforce_tensor_spec(pf_input.to_numpy(), tensor_spec)
        elif isinstance(pf_input, np.ndarray) or _is_sparse_matrix(pf_input):
            new_pf_input = _enforce_tensor_spec(pf_input, tensor_spec)
        else:
            raise MlflowException(f'This model contains a tensor-based model signature with no input names, which suggests a numpy array input or a pandas dataframe input with proper column values, but an input of type {type(pf_input)} was found.', error_code=INVALID_PARAMETER_VALUE)
    return new_pf_input