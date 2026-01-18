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
def _find_app(app_name: str) -> str:
    apps = get_entry_points('mlflow.app')
    for app in apps:
        if app.name == app_name:
            return app.value
    raise MlflowException(f"Failed to find app '{app_name}'. Available apps: {[a.name for a in apps]}")