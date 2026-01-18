import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
class SMTP_SSL(SMTP):
    """ This is a subclass derived from SMTP that connects over an SSL
        encrypted socket (to use this class you need a socket module that was
        compiled with SSL support). If host is not specified, '' (the local
        host) is used. If port is omitted, the standard SMTP-over-SSL port
        (465) is used.  local_hostname and source_address have the same meaning
        as they do in the SMTP class.  keyfile and certfile are also optional -
        they can contain a PEM formatted private key and certificate chain file
        for the SSL connection. context also optional, can contain a
        SSLContext, and is an alternative to keyfile and certfile; If it is
        specified both keyfile and certfile must be None.

        """
    default_port = SMTP_SSL_PORT

    def __init__(self, host='', port=0, local_hostname=None, keyfile=None, certfile=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, context=None):
        if context is not None and keyfile is not None:
            raise ValueError('context and keyfile arguments are mutually exclusive')
        if context is not None and certfile is not None:
            raise ValueError('context and certfile arguments are mutually exclusive')
        if keyfile is not None or certfile is not None:
            import warnings
            warnings.warn('keyfile and certfile are deprecated, use a custom context instead', DeprecationWarning, 2)
        self.keyfile = keyfile
        self.certfile = certfile
        if context is None:
            context = ssl._create_stdlib_context(certfile=certfile, keyfile=keyfile)
        self.context = context
        SMTP.__init__(self, host, port, local_hostname, timeout, source_address)

    def _get_socket(self, host, port, timeout):
        if self.debuglevel > 0:
            self._print_debug('connect:', (host, port))
        new_socket = super()._get_socket(host, port, timeout)
        new_socket = self.context.wrap_socket(new_socket, server_hostname=self._host)
        return new_socket