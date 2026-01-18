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
def _infer_array_datatype(data: Union[List, np.ndarray]) -> Optional[Array]:
    """Infer schema from an array. This tries to infer type if there is at least one
    non-null item in the list, assuming the list has a homogeneous type. However,
    if the list is empty or all items are null, returns None as a sign of undetermined.

    E.g.
        ["a", "b"] => Array(string)
        ["a", None] => Array(string)
        [["a", "b"], []] => Array(Array(string))
        [] => None

    Args:
        data: data to infer from.

    Returns:
        Array(dtype) or None if undetermined
    """
    result = None
    for item in data:
        if _is_none_or_nan(item):
            continue
        dtype = _infer_datatype(item)
        if dtype is None:
            continue
        if result is None:
            result = Array(dtype)
        elif isinstance(result.dtype, (Array, Object, Map)):
            try:
                result = Array(result.dtype._merge(dtype))
            except MlflowException as e:
                raise MlflowException.invalid_parameter_value('Expected all values in list to be of same type') from e
        elif isinstance(result.dtype, DataType):
            if dtype != result.dtype:
                raise MlflowException.invalid_parameter_value('Expected all values in list to be of same type')
        else:
            raise MlflowException.invalid_parameter_value(f'{dtype} is not a valid type for an item of a list or numpy array.')
    return result