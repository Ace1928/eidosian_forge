from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def CreateByteVector(self, x):
    """CreateString writes a byte vector."""
    self.assertNotNested()
    self.nested = True
    if not isinstance(x, compat.binary_types):
        raise TypeError('non-byte vector passed to CreateByteVector')
    self.Prep(N.UOffsetTFlags.bytewidth, len(x) * N.Uint8Flags.bytewidth)
    l = UOffsetTFlags.py_type(len(x))
    self.head = UOffsetTFlags.py_type(self.Head() - l)
    self.Bytes[self.Head():self.Head() + l] = x
    self.vectorNumElems = len(x)
    return self.EndVector()