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
def _read_example_params(mlflow_model: Model, path: str):
    """
    Read params of input_example from a model directory. Returns None if there is no params
    in the input_example or the model was saved without example.
    """
    if mlflow_model.saved_input_example_info is None or mlflow_model.saved_input_example_info.get(EXAMPLE_PARAMS_KEY, None) is None:
        return None
    input_example_dict = _get_mlflow_model_input_example_dict(mlflow_model, path)
    return input_example_dict[EXAMPLE_PARAMS_KEY]