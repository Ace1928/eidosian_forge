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
def _validate_num_items(d, num_items):
    actual_num_items = len(d)
    if actual_num_items != num_items:
        raise MlflowException(f'Expected example to be dict with {num_items} items, got {actual_num_items}', INVALID_PARAMETER_VALUE)