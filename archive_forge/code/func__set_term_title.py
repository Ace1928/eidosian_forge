import os
import sys
import warnings
from shutil import get_terminal_size as _get_terminal_size
def _set_term_title(title):
    """Set terminal title using ctypes to access the Win32 APIs."""
    SetConsoleTitleW(title)