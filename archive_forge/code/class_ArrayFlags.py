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
class ArrayFlags(Type):
    """
    This is the type for `np.ndarray.flags`.
    """

    def __init__(self, arytype):
        self.array_type = arytype
        name = 'ArrayFlags({0})'.format(self.array_type)
        super(ArrayFlags, self).__init__(name)

    @property
    def key(self):
        return self.array_type