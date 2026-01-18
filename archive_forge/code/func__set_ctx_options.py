from __future__ import annotations
import OpenSSL.SSL  # type: ignore[import-untyped]
from cryptography import x509
import logging
import ssl
import typing
from io import BytesIO
from socket import socket as socket_cls
from socket import timeout
from .. import util
def _set_ctx_options(self) -> None:
    self._ctx.set_options(self._options | _openssl_to_ssl_minimum_version[self._minimum_version] | _openssl_to_ssl_maximum_version[self._maximum_version])