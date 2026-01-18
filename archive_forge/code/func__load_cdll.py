import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
from ctypes.util import find_library
from ._ssl_constants import _set_ssl_context_verify_mode
def _load_cdll(name: str, macos10_16_path: str) -> CDLL:
    """Loads a CDLL by name, falling back to known path on 10.16+"""
    try:
        path: str | None
        if _mac_version_info >= (10, 16):
            path = macos10_16_path
        else:
            path = find_library(name)
        if not path:
            raise OSError
        return CDLL(path, use_errno=True)
    except OSError:
        raise ImportError(f'The library {name} failed to load') from None