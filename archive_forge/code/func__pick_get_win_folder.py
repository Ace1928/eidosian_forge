from __future__ import annotations
import ctypes
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING
from .api import PlatformDirsABC
def _pick_get_win_folder() -> Callable[[str], str]:
    if hasattr(ctypes, 'windll'):
        return get_win_folder_via_ctypes
    try:
        import winreg
    except ImportError:
        return get_win_folder_from_env_vars
    else:
        return get_win_folder_from_registry