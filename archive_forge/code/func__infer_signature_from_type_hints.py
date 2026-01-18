import inspect
import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, get_type_hints
import numpy as np
import pandas as pd
from mlflow import environment_variables
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _contains_params, _Example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.types.schema import ParamSchema, Schema
from mlflow.types.utils import _infer_param_schema, _infer_schema, _infer_schema_from_type_hint
from mlflow.utils.uri import append_to_uri_path
def _infer_signature_from_type_hints(func, input_arg_index, input_example=None):
    hints = _extract_type_hints(func, input_arg_index)
    if hints.input is None:
        return None
    params = None
    params_key = 'params'
    if _contains_params(input_example):
        input_example, params = input_example
    input_schema = _infer_schema_from_type_hint(hints.input, input_example) if hints.input else None
    params_schema = _infer_param_schema(params) if params else None
    input_arg_name = _get_arg_names(func)[input_arg_index]
    if input_example:
        inputs = {input_arg_name: input_example}
        if params and params_key in inspect.signature(func).parameters:
            inputs[params_key] = params
        if input_arg_index == 1:
            inputs['context'] = None
        output_example = func(**inputs)
    else:
        output_example = None
    output_schema = _infer_schema_from_type_hint(hints.output, output_example) if hints.output else None
    if not any([input_schema, output_schema, params_schema]):
        return None
    return ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)