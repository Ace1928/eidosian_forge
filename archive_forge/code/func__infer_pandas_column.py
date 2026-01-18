import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def _infer_pandas_column(col: pd.Series) -> DataType:
    if not isinstance(col, pd.Series):
        raise TypeError(f"Expected pandas.Series, got '{type(col)}'.")
    if len(col.values.shape) > 1:
        raise MlflowException(f'Expected 1d array, got array with shape {col.shape}')
    if col.dtype.kind == 'O':
        col = col.infer_objects()
    if col.dtype.kind == 'O':
        try:
            arr_type = _infer_colspec_type(col.to_list())
            return arr_type.dtype
        except Exception as e:
            if pd.api.types.is_string_dtype(col):
                return DataType.string
            raise MlflowException(f'Failed to infer schema for pandas.Series {col}. Error: {e}')
    else:
        return _infer_numpy_dtype(col.dtype)