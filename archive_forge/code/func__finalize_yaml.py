from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def _finalize_yaml(module, available):
    if available and hasattr(module, 'SafeLoader'):
        yaml_load_args['Loader'] = module.SafeLoader