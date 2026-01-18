import inspect
import json
import logging
import os
import shlex
import sys
import traceback
from typing import Dict, NamedTuple, Optional, Tuple
import flask
from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import _log_warning_if_params_not_in_predict_signature
from mlflow.types import ParamSchema, Schema
from mlflow.utils import reraise
from mlflow.utils.annotations import deprecated
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
from mlflow.utils.proto_json_utils import (
from mlflow.version import VERSION
from io import StringIO
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.server.handlers import catch_mlflow_exception
def _handle_serving_error(error_message, error_code, include_traceback=True):
    """
    Logs information about an exception thrown by model inference code that is currently being
    handled and reraises it with the specified error message. The exception stack trace
    is also included in the reraised error message.

    Args:
        error_message: A message for the reraised exception.
        error_code: An appropriate error code for the reraised exception. This should be one of
            the codes listed in the `mlflow.protos.databricks_pb2` proto.
        include_traceback: Whether to include the current traceback in the returned error.
    """
    if include_traceback:
        traceback_buf = StringIO()
        traceback.print_exc(file=traceback_buf)
        traceback_str = traceback_buf.getvalue()
        e = MlflowException(message=error_message, error_code=error_code, stack_trace=traceback_str)
    else:
        e = MlflowException(message=error_message, error_code=error_code)
    reraise(MlflowException, e)