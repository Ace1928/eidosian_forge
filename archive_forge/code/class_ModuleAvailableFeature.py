import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class ModuleAvailableFeature(Feature):
    """This is a feature than describes a module we want to be available.

    Declare the name of the module in __init__(), and then after probing, the
    module will be available as 'self.module'.

    :ivar module: The module if it is available, else None.
    """

    def __init__(self, module_name, ignore_warnings=None):
        super().__init__()
        self.module_name = module_name
        if ignore_warnings is None:
            ignore_warnings = ()
        self.ignore_warnings = ignore_warnings

    def _probe(self):
        sentinel = object()
        module = sys.modules.get(self.module_name, sentinel)
        if module is sentinel:
            with warnings.catch_warnings():
                for warning_category in self.ignore_warnings:
                    warnings.simplefilter('ignore', warning_category)
                try:
                    self._module = importlib.import_module(self.module_name)
                except ImportError:
                    return False
                return True
        else:
            self._module = module
            return True

    @property
    def module(self):
        if self.available():
            return self._module
        return None

    def feature_name(self):
        return self.module_name