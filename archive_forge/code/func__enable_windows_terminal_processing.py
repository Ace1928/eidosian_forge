import ctypes
import platform
import sys
from pprint import pformat as pformat_
from typing import Any
from packaging.version import Version as parse_version
def _enable_windows_terminal_processing() -> bool:
    kernel32 = ctypes.windll.kernel32
    return bool(kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7))