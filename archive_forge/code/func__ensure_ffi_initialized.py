from __future__ import annotations
import os
import sys
import threading
import types
import typing
import warnings
import cryptography
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import _openssl, openssl
from cryptography.hazmat.bindings.openssl._conditional import CONDITIONAL_NAMES
@classmethod
def _ensure_ffi_initialized(cls) -> None:
    with cls._init_lock:
        if not cls._lib_loaded:
            cls.lib = build_conditional_library(_openssl.lib, CONDITIONAL_NAMES)
            cls._lib_loaded = True
            if cls.lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER:
                if not os.environ.get('CRYPTOGRAPHY_OPENSSL_NO_LEGACY'):
                    cls._legacy_provider = cls.lib.OSSL_PROVIDER_load(cls.ffi.NULL, b'legacy')
                    cls._legacy_provider_loaded = cls._legacy_provider != cls.ffi.NULL
                    _legacy_provider_error(cls._legacy_provider_loaded)
                cls._default_provider = cls.lib.OSSL_PROVIDER_load(cls.ffi.NULL, b'default')
                _openssl_assert(cls.lib, cls._default_provider != cls.ffi.NULL)