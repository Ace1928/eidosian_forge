import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
class ConverterLockError(ConverterError):
    """
    Exception raised when an attempt is made to upgrade a locked converter.

    """
    pass