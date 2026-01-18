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
def _get_mlflow_model_input_example_dict(mlflow_model: Model, path: str):
    """
    Args:
        mlflow_model: Model metadata.
        path: Path to the model directory.

    Returns:
        Input example or None if the model has no example.
    """
    if mlflow_model.saved_input_example_info is None:
        return None
    example_type = mlflow_model.saved_input_example_info['type']
    if example_type not in ['dataframe', 'ndarray', 'sparse_matrix_csc', 'sparse_matrix_csr', 'json_object']:
        raise MlflowException(f'This version of mlflow can not load example of type {example_type}')
    path = os.path.join(path, mlflow_model.saved_input_example_info['artifact_path'])
    with open(path) as handle:
        return json.load(handle)