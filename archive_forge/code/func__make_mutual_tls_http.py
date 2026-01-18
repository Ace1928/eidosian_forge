from __future__ import absolute_import
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
def _make_mutual_tls_http(cert, key):
    """Create a mutual TLS HTTP connection with the given client cert and key.
    See https://github.com/urllib3/urllib3/issues/474#issuecomment-253168415

    Args:
        cert (bytes): client certificate in PEM format
        key (bytes): client private key in PEM format

    Returns:
        urllib3.PoolManager: Mutual TLS HTTP connection.

    Raises:
        ImportError: If certifi or pyOpenSSL is not installed.
        OpenSSL.crypto.Error: If the cert or key is invalid.
    """
    import certifi
    from OpenSSL import crypto
    import urllib3.contrib.pyopenssl
    urllib3.contrib.pyopenssl.inject_into_urllib3()
    ctx = urllib3.util.ssl_.create_urllib3_context()
    ctx.load_verify_locations(cafile=certifi.where())
    pkey = crypto.load_privatekey(crypto.FILETYPE_PEM, key)
    x509 = crypto.load_certificate(crypto.FILETYPE_PEM, cert)
    ctx._ctx.use_certificate(x509)
    ctx._ctx.use_privatekey(pkey)
    http = urllib3.PoolManager(ssl_context=ctx)
    return http