from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class PyperclipWindowsException(PyperclipException):
    """
    Exception raised when clipboard functionality is unsupported by Windows.

    Access to the clipboard handle would be denied due to some other
    window process is accessing it.
    """

    def __init__(self, message: str) -> None:
        message += f' ({ctypes.WinError()})'
        super().__init__(message)