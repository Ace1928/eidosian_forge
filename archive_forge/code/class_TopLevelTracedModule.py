import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
class TopLevelTracedModule(TracedModule):
    forward: Callable[..., Any] = _CachedForward()

    def _reconstruct(self, cpp_module):
        """
        Re-construct an instance of TopLevelTracedModule using an instance of a C++ module.

        Args:
            cpp_module: The C++ module that this TopLevelTracedModule will be rebuilt around.
        """
        self.__dict__['_actual_script_module']._reconstruct(cpp_module)