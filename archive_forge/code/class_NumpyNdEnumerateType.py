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
class NumpyNdEnumerateType(SimpleIteratorType):
    """
    Type class for `np.ndenumerate()` objects.
    """

    def __init__(self, arrty):
        from . import Tuple, UniTuple, intp
        self.array_type = arrty
        yield_type = Tuple((UniTuple(intp, arrty.ndim), arrty.dtype))
        name = 'ndenumerate({arrayty})'.format(arrayty=arrty)
        super(NumpyNdEnumerateType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.array_type