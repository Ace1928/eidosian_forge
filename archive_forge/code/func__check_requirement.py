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
def _check_requirement(self) -> None:
    assert self.requirement
    try:
        pkg_resources.require(self.requirement)
        self.available = True
        self.message = f'Requirement {self.requirement!r} met'
    except Exception as ex:
        self.available = False
        self.message = f'{ex.__class__.__name__}: {ex}. HINT: Try running `pip install -U {self.requirement!r}`'
        req_include_version = any((c in self.requirement for c in '=<>'))
        if not req_include_version or self.module is not None:
            module = self.requirement if self.module is None else self.module
            self.available = module_available(module)
            if self.available:
                self.message = f'Module {module!r} available'