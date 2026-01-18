import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def _create_function_from_code_obj(fcode, func_env, func_arg, func_clo, glbls):
    """
    Creates a function from a code object. Args:
    * fcode - the code object
    * func_env - string for the freevar placeholders
    * func_arg - string for the function args (e.g. "a, b, c, d=None")
    * func_clo - string for the closure args
    * glbls - the function globals
    """
    sanitized_co_name = fcode.co_name.replace('<', '_').replace('>', '_')
    func_text = f'def closure():\n{func_env}\n\tdef {sanitized_co_name}({func_arg}):\n\t\treturn ({func_clo})\n\treturn {sanitized_co_name}'
    loc = {}
    exec(func_text, glbls, loc)
    f = loc['closure']()
    f.__code__ = fcode
    f.__name__ = fcode.co_name
    return f