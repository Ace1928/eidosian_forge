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
def _infer_datatype(data: Any) -> Union[DataType, Array, Object, Map]:
    if isinstance(data, dict):
        properties = []
        for k, v in data.items():
            dtype = _infer_datatype(v)
            if dtype is None:
                raise MlflowException('Dictionary value must not be an empty list.')
            properties.append(Property(name=k, dtype=dtype))
        return Object(properties=properties)
    if isinstance(data, (list, np.ndarray)):
        return _infer_array_datatype(data)
    return _infer_scalar_datatype(data)