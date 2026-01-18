import ast
import collections
import inspect
import linecache
import numbers
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings
import types
import numpy
from cupy_backends.cuda.api import runtime
from cupy._core._codeblock import CodeBlock, _CodeType
from cupy._core import _kernel
from cupy._core._dtype import _raise_if_invalid_cast
from cupyx import jit
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit import _builtin_funcs
from cupyx.jit import _interface
def _transpile_function_internal(func, name, attributes, mode, consts, in_types, ret_type, generated):
    consts = dict([(k, Constant(v)) for k, v in consts.items()])
    if not isinstance(func, ast.FunctionDef):
        raise NotImplementedError('Not supported: {}'.format(type(func)))
    if len(func.decorator_list) > 0:
        if sys.version_info >= (3, 9):
            for deco in func.decorator_list:
                deco_code = ast.unparse(deco)
                if not any((word in deco_code for word in ['rawkernel', 'vectorize'])):
                    warnings.warn(f'Decorator {deco_code} may not supported in JIT.', RuntimeWarning)
    arguments = func.args
    if arguments.vararg is not None:
        raise NotImplementedError('`*args` is not supported currently.')
    if len(arguments.kwonlyargs) > 0:
        raise NotImplementedError('keyword only arguments are not supported currently .')
    if arguments.kwarg is not None:
        raise NotImplementedError('`**kwargs` is not supported currently.')
    if len(arguments.defaults) > 0:
        raise NotImplementedError('Default values are not supported currently.')
    args = [arg.arg for arg in arguments.args]
    if len(args) != len(in_types):
        raise TypeError(f'{name}() takes {len(args)} positional arguments but {len(in_types)} were given.')
    params = dict([(x, Data(x, t)) for x, t in zip(args, in_types)])
    env = Environment(mode, consts, params, ret_type, generated)
    body = _transpile_stmts(func.body, True, env)
    params_s = ', '.join([t.declvar(x, None) for x, t in zip(args, in_types)])
    local_vars = [v.ctype.declvar(n, None) + ';' for n, v in env.decls.items()]
    if env.ret_type is None:
        env.ret_type = _cuda_types.void
    head = f'{attributes} {env.ret_type} {name}({params_s})'
    code = CodeBlock(head, local_vars + body)
    return (str(code), env)