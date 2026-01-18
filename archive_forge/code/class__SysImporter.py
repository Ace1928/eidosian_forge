import importlib
from abc import ABC, abstractmethod
from pickle import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
from ._mangling import demangle, get_mangle_prefix, is_mangled
class _SysImporter(Importer):
    """An importer that implements the default behavior of Python."""

    def import_module(self, module_name: str):
        return importlib.import_module(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        return _pickle_whichmodule(obj, name)