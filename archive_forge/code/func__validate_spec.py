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
def _validate_spec(spec):
    for k, v in spec.items():
        if not isinstance(k, str):
            raise TypeError('spec keys should be strings, got %r' % (k,))
        if not isinstance(v, types.Type):
            raise TypeError('spec values should be Numba type instances, got %r' % (v,))