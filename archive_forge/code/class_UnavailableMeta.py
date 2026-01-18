from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
class UnavailableMeta(type):

    def __getattr__(cls, name):
        raise DeferredImportError(unavailable_module._moduleunavailable_message(f"The class attribute '{cls.__name__}.{name}' is not available because a needed optional dependency was not found"))