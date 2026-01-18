from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def _pyutilib_importer():
    importlib.import_module('pyutilib.component')
    return importlib.import_module('pyutilib')