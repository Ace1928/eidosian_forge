import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
class CERT_CHAIN_POLICY_STATUS(Structure):
    _fields_ = (('cbSize', DWORD), ('dwError', DWORD), ('lChainIndex', LONG), ('lElementIndex', LONG), ('pvExtraPolicyStatus', c_void_p))