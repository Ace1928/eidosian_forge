from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def ForceDefaults(self, forceDefaults):
    """
        In order to save space, fields that are set to their default value
        don't get serialized into the buffer. Forcing defaults provides a
        way to manually disable this optimization. When set to `True`, will
        always serialize default values.
        """
    self.forceDefaults = forceDefaults