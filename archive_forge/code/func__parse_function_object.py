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
def _parse_function_object(func):
    """Returns the tuple of ``ast.FunctionDef`` object and the source string
    for the given callable ``func``.

    ``func`` can be a ``def`` function or a ``lambda`` expression.

    The source is returned only for informational purposes (i.e., rendering
    an exception message in case of an error).
    """
    if not callable(func):
        raise ValueError('`func` must be a callable object.')
    try:
        filename = inspect.getsourcefile(func)
    except TypeError:
        filename = None
    if filename == '<stdin>':
        raise RuntimeError(f'JIT needs access to the Python source code for {func} but it cannot be retrieved within the Python interactive interpreter. Consider using IPython instead.')
    if func.__name__ != '<lambda>':
        lines, _ = inspect.getsourcelines(func)
        num_indent = len(lines[0]) - len(lines[0].lstrip())
        source = ''.join([line.replace(' ' * num_indent, '', 1) for line in lines])
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        return (tree.body[0], source)
    if filename is None:
        raise ValueError(f'JIT needs access to Python source code for {func} but could not be located.\n(hint: it is likely you passed a built-in function or method)')
    full_source = ''.join(linecache.getlines(filename))
    source, start_line = inspect.getsourcelines(func)
    end_line = start_line + len(source)
    source = ''.join(source)
    tree = ast.parse(full_source)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda) and start_line <= node.lineno < end_line]
    if len(nodes) > 1:
        raise ValueError(f'Multiple callables are found near the definition of {func}, and JIT could not identify the source code for it.')
    node = nodes[0]
    return (ast.FunctionDef(name='_lambda_kernel', args=node.args, body=[ast.Return(node.body)], decorator_list=[], returns=None, type_comment=None), source)