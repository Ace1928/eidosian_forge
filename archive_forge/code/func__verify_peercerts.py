import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
def _verify_peercerts(sock_or_sslobj: ssl.SSLSocket | ssl.SSLObject, server_hostname: str | None) -> None:
    """
    Verifies the peer certificates from an SSLSocket or SSLObject
    against the certificates in the OS trust store.
    """
    sslobj: ssl.SSLObject = sock_or_sslobj
    try:
        while not hasattr(sslobj, 'get_unverified_chain'):
            sslobj = sslobj._sslobj
    except AttributeError:
        pass
    unverified_chain: typing.Sequence[_ssl.Certificate] = sslobj.get_unverified_chain() or ()
    cert_bytes = [cert.public_bytes(_ssl.ENCODING_DER) for cert in unverified_chain]
    _verify_peercerts_impl(sock_or_sslobj.context, cert_bytes, server_hostname=server_hostname)