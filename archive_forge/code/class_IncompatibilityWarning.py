from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class IncompatibilityWarning(Warning):
    """
    Warning raised when trying to use where criteria on an incompatible HDF5 file.
    """