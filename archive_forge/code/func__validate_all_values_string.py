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
def _validate_all_values_string(d):
    values = list(d.values())
    if not _is_all_string(values):
        raise MlflowException(f'Expected example to be dict with string values, got {values}', INVALID_PARAMETER_VALUE)