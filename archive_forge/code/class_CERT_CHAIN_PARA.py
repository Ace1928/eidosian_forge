import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
class CERT_CHAIN_PARA(Structure):
    _fields_ = (('cbSize', DWORD), ('RequestedUsage', CERT_USAGE_MATCH), ('RequestedIssuancePolicy', CERT_USAGE_MATCH), ('dwUrlRetrievalTimeout', DWORD), ('fCheckRevocationFreshnessTime', BOOL), ('dwRevocationFreshnessTime', DWORD), ('pftCacheResync', LPFILETIME), ('pStrongSignPara', c_void_p), ('dwStrongSignFlags', DWORD))