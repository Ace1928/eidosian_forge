import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
class SSL_EXTRA_CERT_CHAIN_POLICY_PARA(Structure):
    _fields_ = (('cbSize', DWORD), ('dwAuthType', DWORD), ('fdwChecks', DWORD), ('pwszServerName', LPCWSTR))