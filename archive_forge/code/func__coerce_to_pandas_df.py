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
def _coerce_to_pandas_df(input_ex):
    if isinstance(input_ex, dict):
        if all((isinstance(x, str) or (isinstance(x, list) and all((_is_scalar(y) for y in x))) for x in input_ex.values())):
            _logger.info('We convert input dictionaries to pandas DataFrames such that each key represents a column, collectively constituting a single row of data. If you would like to save data as multiple rows, please convert your data to a pandas DataFrame before passing to input_example.')
        input_ex = pd.DataFrame([input_ex])
    elif np.isscalar(input_ex):
        input_ex = pd.DataFrame([input_ex])
    elif not isinstance(input_ex, pd.DataFrame):
        input_ex = None
    return input_ex