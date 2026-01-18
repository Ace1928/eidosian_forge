from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def Slot(self, slotnum):
    """
        Slot sets the vtable key `voffset` to the current location in the
        buffer.

        """
    self.assertNested()
    self.current_vtable[slotnum] = self.Offset()