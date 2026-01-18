from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
class StructIsNotInlineError(RuntimeError):
    """
    Error caused by using a Builder to write a Struct at a location that
    is not the current Offset.
    """
    pass