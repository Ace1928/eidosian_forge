from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
class BuilderSizeError(RuntimeError):
    """
    Error caused by causing a Builder to exceed the hardcoded limit of 2
    gigabytes.
    """
    pass