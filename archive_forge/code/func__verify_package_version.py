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
def _verify_package_version(version: str) -> None:
    so_package_version = _openssl.ffi.string(_openssl.lib.CRYPTOGRAPHY_PACKAGE_VERSION)
    if version.encode('ascii') != so_package_version:
        raise ImportError('The version of cryptography does not match the loaded shared object. This can happen if you have multiple copies of cryptography installed in your Python path. Please try creating a new virtual environment to resolve this issue. Loaded python version: {}, shared object version: {}'.format(version, so_package_version))
    _openssl_assert(_openssl.lib, _openssl.lib.OpenSSL_version_num() == openssl.openssl_version())