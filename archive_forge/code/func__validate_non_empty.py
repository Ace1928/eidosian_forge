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
def _validate_non_empty(examples):
    num_items = len(examples)
    if num_items == 0:
        raise MlflowException(f'Expected examples to be non-empty list, got {num_items}', INVALID_PARAMETER_VALUE)