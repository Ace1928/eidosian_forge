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
def _infer_scalar_datatype(data) -> DataType:
    if DataType.is_boolean(data):
        return DataType.boolean
    if DataType.is_long(data):
        return DataType.long
    if DataType.is_integer(data):
        return DataType.integer
    if DataType.is_double(data):
        return DataType.double
    if DataType.is_float(data):
        return DataType.float
    if DataType.is_string(data):
        return DataType.string
    if DataType.is_binary(data):
        return DataType.binary
    if DataType.is_datetime(data):
        return DataType.datetime
    raise MlflowException.invalid_parameter_value(f'Data {data} is not one of the supported DataType')