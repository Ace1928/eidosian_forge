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
def _legacy_provider_error(loaded: bool) -> None:
    if not loaded:
        raise RuntimeError("OpenSSL 3.0's legacy provider failed to load. This is a fatal error by default, but cryptography supports running without legacy algorithms by setting the environment variable CRYPTOGRAPHY_OPENSSL_NO_LEGACY. If you did not expect this error, you have likely made a mistake with your OpenSSL configuration.")