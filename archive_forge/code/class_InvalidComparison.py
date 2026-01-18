from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class InvalidComparison(Exception):
    """
    Exception is raised by _validate_comparison_value to indicate an invalid comparison.

    Notes
    -----
    This is an internal error.
    """