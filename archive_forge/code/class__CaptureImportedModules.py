import argparse
import builtins
import functools
import importlib
import json
import os
import sys
import mlflow
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.pyfunc import MAIN
from mlflow.utils._spark_utils import _prepare_subprocess_environ_for_creating_local_spark_session
from mlflow.utils.exception_utils import get_stacktrace
from mlflow.utils.file_utils import write_to
from mlflow.utils.requirements_utils import (
class _CaptureImportedModules:
    """
    A context manager to capture imported modules by temporarily applying a patch to
    `builtins.__import__` and `importlib.import_module`.
    """

    def __init__(self):
        self.imported_modules = set()
        self.original_import = None
        self.original_import_module = None

    def _wrap_import(self, original):

        @functools.wraps(original)
        def wrapper(name, globals=None, locals=None, fromlist=(), level=0):
            is_absolute_import = level == 0
            if is_absolute_import:
                self._record_imported_module(name)
            return original(name, globals, locals, fromlist, level)
        return wrapper

    def _wrap_import_module(self, original):

        @functools.wraps(original)
        def wrapper(name, *args, **kwargs):
            self._record_imported_module(name)
            return original(name, *args, **kwargs)
        return wrapper

    def _record_imported_module(self, full_module_name):
        if full_module_name.startswith('_') or full_module_name == 'databricks':
            return
        top_level_module = _get_top_level_module(full_module_name)
        second_level_module = _get_second_level_module(full_module_name)
        if top_level_module == 'databricks':
            if second_level_module in DATABRICKS_MODULES_TO_PACKAGES:
                self.imported_modules.add(second_level_module)
                return
            for databricks_module in DATABRICKS_MODULES_TO_PACKAGES:
                if full_module_name.startswith(databricks_module):
                    self.imported_modules.add(databricks_module)
                    return
        if top_level_module == 'mlflow':
            if second_level_module in MLFLOW_MODULES_TO_PACKAGES:
                self.imported_modules.add(second_level_module)
                return
        self.imported_modules.add(top_level_module)

    def __enter__(self):
        self.original_import = builtins.__import__
        self.original_import_module = importlib.import_module
        builtins.__import__ = self._wrap_import(self.original_import)
        importlib.import_module = self._wrap_import_module(self.original_import_module)
        return self

    def __exit__(self, *_, **__):
        builtins.__import__ = self.original_import
        importlib.import_module = self.original_import_module