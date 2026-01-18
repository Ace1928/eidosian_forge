import importlib
import importlib.metadata
import os
import shlex
import sys
import textwrap
import types
from flask import Flask, Response, send_from_directory
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.handlers import (
from mlflow.utils.os import get_entry_points, is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.version import VERSION
def _is_factory(app: str) -> bool:
    """
    Returns True if the given app is a factory function, False otherwise.

    Args:
        app: The app to check, e.g. "mlflow.server.app:app
    """
    module, obj_name = app.rsplit(':', 1)
    mod = importlib.import_module(module)
    obj = getattr(mod, obj_name)
    return isinstance(obj, types.FunctionType)