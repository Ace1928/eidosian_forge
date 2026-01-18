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
def _extract_type_hints(f, input_arg_index):
    """
    Extract type hints from a function.

    Args:
        f: Function to extract type hints from.
        input_arg_index: Index of the function argument that corresponds to the model input.

    Returns:
        A `_TypeHints` object containing the input and output type hints.
    """
    if not hasattr(f, '__annotations__') and hasattr(f, '__call__'):
        return _extract_type_hints(f.__call__, input_arg_index)
    if f.__annotations__ == {}:
        return _TypeHints()
    arg_names = _get_arg_names(f)
    if len(arg_names) - 1 < input_arg_index:
        raise MlflowException.invalid_parameter_value(f'The specified input argument index ({input_arg_index}) is out of range for the function signature: {{}}'.format(input_arg_index, arg_names))
    arg_name = _get_arg_names(f)[input_arg_index]
    try:
        hints = get_type_hints(f)
    except TypeError:
        hints = {}
        for arg in [arg_name, 'return']:
            if (hint_str := f.__annotations__.get(arg, None)):
                if (hint := _infer_hint_from_str(hint_str)):
                    hints[arg] = hint
                else:
                    _logger.info('Unsupported type hint: %s, skipping schema inference', hint_str)
    except Exception as e:
        _logger.warning('Failed to extract type hints from function %s: %s', f.__name__, repr(e))
        return _TypeHints()
    return _TypeHints(hints.get(arg_name), hints.get('return'))