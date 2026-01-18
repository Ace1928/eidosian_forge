import functools
import importlib
import os
import warnings
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, List, Optional, TypeVar
import pkg_resources
from packaging.requirements import Requirement
from packaging.version import Version
from typing_extensions import ParamSpec
class ModuleAvailableCache(RequirementCache):
    """Boolean-like class for check of module availability.

    >>> ModuleAvailableCache("torch")
    Module 'torch' available
    >>> bool(ModuleAvailableCache("torch.utils"))
    True
    >>> bool(ModuleAvailableCache("unknown_package"))
    False
    >>> bool(ModuleAvailableCache("unknown.module.path"))
    False

    """

    def __init__(self, module: str) -> None:
        warnings.warn('`ModuleAvailableCache` is a special case of `RequirementCache`. Please use `RequirementCache(module=...)` instead.', DeprecationWarning, stacklevel=4)
        super().__init__(module=module)