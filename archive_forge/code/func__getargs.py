import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
def _getargs(fn_sig):
    """
    Returns list of positional and keyword argument names in order.
    """
    params = fn_sig.parameters
    args = []
    for k, v in params.items():
        if v.kind & v.POSITIONAL_OR_KEYWORD == v.POSITIONAL_OR_KEYWORD:
            args.append(k)
        else:
            msg = '%s argument type unsupported in jitclass' % v.kind
            raise errors.UnsupportedError(msg)
    return args