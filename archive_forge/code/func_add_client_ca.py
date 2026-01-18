import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def add_client_ca(self, certificate_authority):
    """
        Add the CA certificate to the list of preferred signers for this
        context.

        The list of certificate authorities will be sent to the client when the
        server requests a client certificate.

        :param certificate_authority: certificate authority's X509 certificate.
        :return: None

        .. versionadded:: 0.10
        """
    if not isinstance(certificate_authority, X509):
        raise TypeError('certificate_authority must be an X509 instance')
    add_result = _lib.SSL_CTX_add_client_CA(self._context, certificate_authority._x509)
    _openssl_assert(add_result == 1)