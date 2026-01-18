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
def _get_pyfunc_supported_input_types():
    import mlflow.models.utils as base_module
    supported_input_types = []
    for input_type in get_args(base_module.PyFuncInput):
        if isinstance(input_type, type):
            supported_input_types.append(input_type)
        elif isinstance(input_type, ForwardRef):
            name = input_type.__forward_arg__
            if hasattr(base_module, name):
                cls = getattr(base_module, name)
                supported_input_types.append(cls)
        else:
            supported_input_types.append(get_origin(input_type))
    return tuple(supported_input_types)