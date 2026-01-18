from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
class UnavailableBase(metaclass=UnavailableMeta):

    def __new__(cls, *args, **kwargs):
        raise DeferredImportError(unavailable_module._moduleunavailable_message(f"The class '{cls.__name__}' cannot be created because a needed optional dependency was not found"))