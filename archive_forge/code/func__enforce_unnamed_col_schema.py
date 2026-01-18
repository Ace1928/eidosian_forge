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
def _enforce_unnamed_col_schema(pf_input: pd.DataFrame, input_schema: Schema):
    """Enforce the input columns conform to the model's column-based signature."""
    input_names = pf_input.columns[:len(input_schema.inputs)]
    input_types = input_schema.input_types()
    new_pf_input = {}
    for i, x in enumerate(input_names):
        if isinstance(input_types[i], DataType):
            new_pf_input[x] = _enforce_mlflow_datatype(x, pf_input[x], input_types[i])
        else:
            new_pf_input[x] = pd.Series([_enforce_type(obj, input_types[i]) for obj in pf_input[x]], name=x)
    return pd.DataFrame(new_pf_input)