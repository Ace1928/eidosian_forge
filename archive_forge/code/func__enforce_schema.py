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
def _enforce_schema(pf_input: PyFuncInput, input_schema: Schema, flavor: Optional[str]=None):
    """
    Enforces the provided input matches the model's input schema,

    For signatures with input names, we check there are no missing inputs and reorder the inputs to
    match the ordering declared in schema if necessary. Any extra columns are ignored.

    For column-based signatures, we make sure the types of the input match the type specified in
    the schema or if it can be safely converted to match the input schema.

    For Pyspark DataFrame inputs, MLflow casts a sample of the PySpark DataFrame into a Pandas
    DataFrame. MLflow will only enforce the schema on a subset of the data rows.

    For tensor-based signatures, we make sure the shape and type of the input matches the shape
    and type specified in model's input schema.
    """

    def _is_scalar(x):
        return np.isscalar(x) or x is None
    original_pf_input = pf_input
    if isinstance(pf_input, pd.Series):
        pf_input = pd.DataFrame(pf_input)
    if not input_schema.is_tensor_spec():
        if np.isscalar(pf_input):
            pf_input = pd.DataFrame([pf_input])
        elif isinstance(pf_input, dict):
            if any((isinstance(col_spec.type, (Array, Object)) for col_spec in input_schema.inputs)) or all((_is_scalar(value) or (isinstance(value, list) and all((isinstance(item, str) for item in value))) for value in pf_input.values())):
                pf_input = pd.DataFrame([pf_input])
            else:
                try:
                    if all((isinstance(value, np.ndarray) and value.dtype.type == np.str_ and (value.size == 1) and (value.shape == ()) for value in pf_input.values())):
                        pf_input = pd.DataFrame([pf_input])
                    elif any((isinstance(value, np.ndarray) and value.ndim > 1 for value in pf_input.values())):
                        pf_input = pd.DataFrame({key: value.tolist() if isinstance(value, np.ndarray) and value.ndim > 1 else value for key, value in pf_input.items()})
                    else:
                        pf_input = pd.DataFrame(pf_input)
                except Exception as e:
                    raise MlflowException(f'This model contains a column-based signature, which suggests a DataFrame input. There was an error casting the input data to a DataFrame: {e}')
        elif isinstance(pf_input, (list, np.ndarray, pd.Series)):
            pf_input = pd.DataFrame(pf_input)
        elif HAS_PYSPARK and isinstance(pf_input, SparkDataFrame):
            pf_input = pf_input.limit(10).toPandas()
            for field in original_pf_input.schema.fields:
                if isinstance(field.dataType, (StructType, ArrayType)):
                    pf_input[field.name] = pf_input[field.name].apply(lambda row: convert_complex_types_pyspark_to_pandas(row, field.dataType))
        if not isinstance(pf_input, pd.DataFrame):
            raise MlflowException(f'Expected input to be DataFrame. Found: {type(pf_input).__name__}')
    if input_schema.has_input_names():
        input_names = input_schema.required_input_names()
        optional_names = input_schema.optional_input_names()
        expected_required_cols = set(input_names)
        actual_cols = set()
        optional_cols = set(optional_names)
        if len(expected_required_cols) == 1 and isinstance(pf_input, np.ndarray):
            pf_input = {input_names[0]: pf_input}
            actual_cols = expected_required_cols
        elif isinstance(pf_input, pd.DataFrame):
            actual_cols = set(pf_input.columns)
        elif isinstance(pf_input, dict):
            actual_cols = set(pf_input.keys())
        missing_cols = expected_required_cols - actual_cols
        extra_cols = actual_cols - expected_required_cols - optional_cols
        missing_cols = [c for c in input_names if c in missing_cols]
        extra_cols = [c for c in actual_cols if c in extra_cols]
        if missing_cols:
            message = f'Model is missing inputs {missing_cols}.'
            if extra_cols:
                message += f' Note that there were extra inputs: {extra_cols}'
            raise MlflowException(message)
    elif not input_schema.is_tensor_spec():
        num_actual_columns = len(pf_input.columns)
        if num_actual_columns < len(input_schema.inputs):
            raise MlflowException('Model inference is missing inputs. The model signature declares {} inputs  but the provided value only has {} inputs. Note: the inputs were not named in the signature so we can only verify their count.'.format(len(input_schema.inputs), num_actual_columns))
    if input_schema.is_tensor_spec():
        return _enforce_tensor_schema(pf_input, input_schema)
    elif HAS_PYSPARK and isinstance(original_pf_input, SparkDataFrame):
        return _enforce_pyspark_dataframe_schema(original_pf_input, pf_input, input_schema, flavor=flavor)
    else:
        return _enforce_named_col_schema(pf_input, input_schema) if input_schema.has_input_names() else _enforce_unnamed_col_schema(pf_input, input_schema)