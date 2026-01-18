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
def _enforce_named_col_schema(pf_input: pd.DataFrame, input_schema: Schema):
    """Enforce the input columns conform to the model's column-based signature."""
    input_names = input_schema.input_names()
    input_dict = input_schema.input_dict()
    new_pf_input = {}
    for name in input_names:
        input_type = input_dict[name].type
        required = input_dict[name].required
        if name not in pf_input:
            if required:
                raise MlflowException(f"The input column '{name}' is required by the model signature but missing from the input data.")
            else:
                continue
        if isinstance(input_type, DataType):
            new_pf_input[name] = _enforce_mlflow_datatype(name, pf_input[name], input_type)
        else:
            new_pf_input[name] = pd.Series([_enforce_type(obj, input_type, required) for obj in pf_input[name]], name=name)
    return pd.DataFrame(new_pf_input)