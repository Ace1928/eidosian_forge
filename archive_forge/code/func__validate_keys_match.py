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
def _validate_keys_match(d, expected_keys):
    if d.keys() != expected_keys:
        raise MlflowException(f'Expected example to be dict with keys {list(expected_keys)}, got {list(d.keys())}', INVALID_PARAMETER_VALUE)