import importlib
import logging
import os
import types
from pathlib import Path
import torch
class _LazyImporter(types.ModuleType):
    """Lazily import module/extension."""

    def __init__(self, name, import_func):
        super().__init__(name)
        self.import_func = import_func
        self.module = None

    def __getattr__(self, item):
        self._import_once()
        return getattr(self.module, item)

    def __repr__(self):
        if self.module is None:
            return f'''<module '{self.__module__}.{self.__class__.__name__}("{self.name}")'>'''
        return repr(self.module)

    def __dir__(self):
        self._import_once()
        return dir(self.module)

    def _import_once(self):
        if self.module is None:
            self.module = self.import_func()
            self.__dict__.update(self.module.__dict__)

    def is_available(self):
        try:
            self._import_once()
        except Exception:
            return False
        return True