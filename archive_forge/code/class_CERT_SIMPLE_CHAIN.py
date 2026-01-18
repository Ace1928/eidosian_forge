import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
class CERT_SIMPLE_CHAIN(Structure):
    _fields_ = (('cbSize', DWORD), ('TrustStatus', CERT_TRUST_STATUS), ('cElement', DWORD), ('rgpElement', POINTER(PCERT_CHAIN_ELEMENT)), ('pTrustListInfo', c_void_p), ('fHasRevocationFreshnessTime', BOOL), ('dwRevocationFreshnessTime', DWORD))