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
def _wrap_exec_module(self, method: Any) -> Any:

    def exec_module(module: ModuleType) -> None:
        assert module.__name__ == self.old_module_name, f'DeprecatedModuleLoader for {self.old_module_name} was asked to load {module.__name__}'
        if self.new_module_name in sys.modules:
            sys.modules[self.old_module_name] = sys.modules[self.new_module_name]
            return
        sys.modules[self.old_module_name] = module
        sys.modules[self.new_module_name] = module
        try:
            return method(module)
        except BaseException:
            del sys.modules[self.new_module_name]
            del sys.modules[self.old_module_name]
            raise
    return exec_module