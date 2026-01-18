import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
class DeprecatedModuleFinder(importlib.abc.MetaPathFinder):
    """A module finder to handle deprecated module references.

    It sends a deprecation warning when a deprecated module is asked to be found.
    It is meant to be used as a wrapper around existing MetaPathFinder instances.

    Args:
        new_module_name: The new module's fully qualified name.
        old_module_name: The deprecated module's fully qualified name.
        deadline: The deprecation deadline.
        broken_module_exception: If specified, an exception to throw if
            the module is found.
    """

    def __init__(self, new_module_name: str, old_module_name: str, deadline: str, broken_module_exception: Optional[BaseException]):
        """An aliasing module finder that uses existing module finders to find a python
        module spec and intercept the execution of matching modules.
        """
        self.new_module_name = new_module_name
        self.old_module_name = old_module_name
        self.deadline = deadline
        self.broken_module_exception = broken_module_exception

    def find_spec(self, fullname: str, path: Any=None, target: Any=None) -> Any:
        """Finds the specification of a module.

        This is an implementation of the importlib.abc.MetaPathFinder.find_spec method.
        See https://docs.python.org/3/library/importlib.html#importlib.abc.MetaPathFinder.

        Args:
            fullname: name of the module.
            path: if presented, this is the parent module's submodule search path.
            target: When passed in, target is a module object that the finder may use to make a more
                educated guess about what spec to return. We don't use it here, just pass it along
                to the wrapped finder.
        """
        if fullname != self.old_module_name and (not fullname.startswith(self.old_module_name + '.')):
            return None
        if self.broken_module_exception is not None:
            raise self.broken_module_exception
        _deduped_module_warn_or_error(self.old_module_name, self.new_module_name, self.deadline)
        new_fullname = self.new_module_name + fullname[len(self.old_module_name):]
        spec = importlib.util.find_spec(new_fullname)
        if spec is not None:
            spec.name = fullname
            if getattr(spec.loader, 'name', None) == new_fullname:
                setattr(spec.loader, 'name', fullname)
            spec.loader = DeprecatedModuleLoader(spec.loader, fullname, new_fullname)
        return spec