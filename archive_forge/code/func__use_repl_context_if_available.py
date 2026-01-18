import functools
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar
import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
def _use_repl_context_if_available(name):
    """Creates a decorator to insert a short circuit that returns the specified REPL context
    attribute if it's available.

    Args:
        name: Attribute name (e.g. "apiUrl").

    Returns:
        Decorator to insert the short circuit.
    """

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                from dbruntime.databricks_repl_context import get_context
                context = get_context()
                if context is not None and hasattr(context, name):
                    return getattr(context, name)
            except Exception:
                pass
            return f(*args, **kwargs)
        return wrapper
    return decorator