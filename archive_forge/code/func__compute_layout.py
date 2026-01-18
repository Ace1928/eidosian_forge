import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
@classmethod
def _compute_layout(cls, arrays):
    c = collections.Counter()
    for a in arrays:
        if not isinstance(a, Array):
            continue
        if a.layout in 'CF' and a.ndim == 1:
            c['C'] += 1
            c['F'] += 1
        elif a.ndim >= 1:
            c[a.layout] += 1
    return 'F' if c['F'] > c['C'] else 'C'