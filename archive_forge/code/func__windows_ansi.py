from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
def _windows_ansi() -> bool:
    from ctypes import windll, byref
    from ctypes.wintypes import DWORD
    kernel = windll.kernel32
    stdout = kernel.GetStdHandle(-11)
    mode = DWORD()
    if not kernel.GetConsoleMode(stdout, byref(mode)):
        return False
    return bool(kernel.SetConsoleMode(stdout, mode.value | 4) or os.environ.get('ANSICON'))