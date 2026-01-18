from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PrependSlot(self, flags, o, x, d):
    if x is not None:
        N.enforce_number(x, flags)
    if d is not None:
        N.enforce_number(d, flags)
    if x != d or (self.forceDefaults and d is not None):
        self.Prepend(flags, x)
        self.Slot(o)