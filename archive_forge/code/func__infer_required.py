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
def _infer_required(col) -> bool:
    if isinstance(col, (list, pd.Series)):
        return not any((_is_none_or_nan(x) for x in col))
    return not _is_none_or_nan(col)