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
def _astype_scalar(x: _T, ctype: _cuda_types.Scalar, casting: _CastingType, env: Environment) -> _T:
    if isinstance(x, Constant):
        assert not isinstance(x, Data)
        return Constant(ctype.dtype.type(x.obj))
    if not isinstance(x.ctype, _cuda_types.Scalar):
        raise TypeError(f'{x.code} is not scalar type.')
    from_t = x.ctype.dtype
    to_t = ctype.dtype
    if from_t == to_t:
        return x
    if not numpy.can_cast(from_t.type(0), to_t, casting):
        raise TypeError(f"Cannot cast from '{from_t}' to {to_t} with casting rule {casting}.")
    if from_t.kind == 'c' and to_t.kind != 'c':
        if to_t.kind != 'b':
            warnings.warn('Casting complex values to real discards the imaginary part', numpy.ComplexWarning)
        return Data(f'({ctype})({x.code}.real())', ctype)
    return Data(f'({ctype})({x.code})', ctype)