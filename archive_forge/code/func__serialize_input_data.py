import json
import logging
import os
from datetime import datetime
from io import StringIO
from typing import ForwardRef, get_args, get_origin
from mlflow.exceptions import MlflowException
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data
def _serialize_input_data(input_data, content_type):
    import pandas as pd
    valid_input_types = {_CONTENT_TYPE_CSV: (str, list, dict, pd.DataFrame), _CONTENT_TYPE_JSON: _get_pyfunc_supported_input_types()}.get(content_type)
    if not isinstance(input_data, valid_input_types):
        raise MlflowException.invalid_parameter_value(f"Input data must be one of {valid_input_types} when content type is '{content_type}', but got {type(input_data)}.")
    if isinstance(input_data, str):
        _validate_string(input_data, content_type)
        return input_data
    try:
        if content_type == _CONTENT_TYPE_CSV:
            return pd.DataFrame(input_data).to_csv(index=False)
        else:
            return _serialize_to_json(input_data)
    except Exception as e:
        raise MlflowException.invalid_parameter_value(message=f'Input data could not be serialized to {content_type}.') from e