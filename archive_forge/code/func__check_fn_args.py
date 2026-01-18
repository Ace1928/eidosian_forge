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
def _check_fn_args(node, fn, args):
    if fn.noinline:
        for idx, arg in enumerate(args):
            if not _is_constexpr(arg) and (not _is_triton_scalar(arg)):
                raise UnsupportedLanguageConstruct(fn.src, node, f'Function {fn.__name__} is marked noinline, but was called with non-scalar argument {fn.arg_names[idx]}:{arg}')