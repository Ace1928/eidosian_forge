from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def FinishSizePrefixed(self, rootTable, file_identifier=None):
    """
        Finish finalizes a buffer, pointing to the given `rootTable`,
        with the size prefixed.
        """
    return self.__Finish(rootTable, True, file_identifier=file_identifier)