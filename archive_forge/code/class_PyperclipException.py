from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class PyperclipException(RuntimeError):
    """
    Exception raised when clipboard functionality is unsupported.

    Raised by ``to_clipboard()`` and ``read_clipboard()``.
    """