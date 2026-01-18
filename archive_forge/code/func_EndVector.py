from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def EndVector(self, numElems=None):
    """EndVector writes data necessary to finish vector construction."""
    self.assertNested()
    self.nested = False
    if numElems:
        warnings.warn('numElems is deprecated.', DeprecationWarning, stacklevel=2)
        if numElems != self.vectorNumElems:
            raise EndVectorLengthMismatched()
    self.PlaceUOffsetT(self.vectorNumElems)
    self.vectorNumElems = None
    return self.Offset()