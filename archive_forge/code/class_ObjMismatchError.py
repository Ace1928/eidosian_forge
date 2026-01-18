import importlib
from abc import ABC, abstractmethod
from pickle import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
from ._mangling import demangle, get_mangle_prefix, is_mangled
class ObjMismatchError(Exception):
    """Raised when an importer found a different object with the same name as the user-provided one."""
    pass