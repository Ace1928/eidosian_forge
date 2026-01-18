import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def _define_name_lookup(self):

    def local_lookup(name: str, absent):
        value = self.lscope.get(name, absent)
        if value is not absent and name not in self.local_defs:
            self.global_uses[name] = value
        return value
    absent_marker = object()

    def name_lookup(name: str) -> Any:
        absent = absent_marker
        for lookup_function in (local_lookup, self.gscope.get, self.builtin_namespace.get):
            value = lookup_function(name, absent)
            if value is not absent:
                return value
        raise NameError(f'{name} is not defined')
    return name_lookup