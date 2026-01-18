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
def _save_example(mlflow_model: Model, input_example: ModelInputExample, path: str, no_conversion=False):
    """
    Saves example to a file on the given path and updates passed Model with example metadata.

    The metadata is a dictionary with the following fields:
      - 'artifact_path': example path relative to the model directory.
      - 'type': Type of example. Currently the supported values are 'dataframe' and 'ndarray'
      -  One of the following metadata based on the `type`:
            - 'pandas_orient': Used to store dataframes. Determines the json encoding for dataframe
                               examples in terms of pandas orient convention. Defaults to 'split'.
            - 'format: Used to store tensors. Determines the standard used to store a tensor input
                       example. MLflow uses a JSON-formatted string representation of TF serving
                       input.

    Args:
        mlflow_model: Model metadata that will get updated with the example metadata.
        path: Where to store the example file. Should be model the model directory.
    """
    if no_conversion:
        example_info = {INPUT_EXAMPLE_PATH: EXAMPLE_FILENAME, 'type': 'json_object'}
        try:
            with open(os.path.join(path, example_info[INPUT_EXAMPLE_PATH]), 'w') as f:
                json.dump(input_example, f, cls=NumpyEncoder)
        except Exception as e:
            raise MlflowException.invalid_parameter_value(f'Failed to save input example. Please make sure the input example is jsonable when no_conversion is True. Got error: {e}') from e
        else:
            mlflow_model.saved_input_example_info = example_info
    else:
        example = _Example(input_example)
        example.save(path)
        mlflow_model.saved_input_example_info = example.info