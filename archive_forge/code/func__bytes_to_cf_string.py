import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
from ctypes.util import find_library
from ._ssl_constants import _set_ssl_context_verify_mode
def _bytes_to_cf_string(value: bytes) -> CFString:
    """
    Given a Python binary data, create a CFString.
    The string must be CFReleased by the caller.
    """
    c_str = ctypes.c_char_p(value)
    cf_str = CoreFoundation.CFStringCreateWithCString(CoreFoundation.kCFAllocatorDefault, c_str, CFConst.kCFStringEncodingUTF8)
    return cf_str