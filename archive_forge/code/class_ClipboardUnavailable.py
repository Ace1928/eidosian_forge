import contextlib
import ctypes
from ctypes import (
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
class ClipboardUnavailable:

    def __call__(self, *args, **kwargs):
        raise PyperclipException(EXCEPT_MSG)

    def __bool__(self) -> bool:
        return False