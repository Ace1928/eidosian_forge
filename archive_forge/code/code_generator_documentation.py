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
 This function:
            called by visit_Assign() & visit_FunctionDef() to store left value (lvalue)
        1. record local defined name (FIXME: should consider control flow)
        2. store tensor in self.lvalue
        