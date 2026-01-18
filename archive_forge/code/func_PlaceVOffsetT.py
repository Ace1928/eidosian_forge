from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PlaceVOffsetT(self, x):
    """PlaceVOffsetT prepends a VOffsetT to the Builder, without checking
        for space.
        """
    N.enforce_number(x, N.VOffsetTFlags)
    self.head = self.head - N.VOffsetTFlags.bytewidth
    encode.Write(packer.voffset, self.Bytes, self.Head(), x)