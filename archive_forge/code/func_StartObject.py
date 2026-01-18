from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def StartObject(self, numfields):
    """StartObject initializes bookkeeping for writing a new object."""
    self.assertNotNested()
    self.current_vtable = [0 for _ in range_func(numfields)]
    self.objectEnd = self.Offset()
    self.nested = True