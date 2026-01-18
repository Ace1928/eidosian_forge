import errno
import re
import socket
import sys
class POP3_SSL(POP3):
    """POP3 client class over SSL connection

        Instantiate with: POP3_SSL(hostname, port=995, keyfile=None, certfile=None,
                                   context=None)

               hostname - the hostname of the pop3 over ssl server
               port - port number
               keyfile - PEM formatted file that contains your private key
               certfile - PEM formatted certificate chain file
               context - a ssl.SSLContext

        See the methods of the parent class POP3 for more documentation.
        """

    def __init__(self, host, port=POP3_SSL_PORT, keyfile=None, certfile=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, context=None):
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
        POP3.__init__(self, host, port, timeout)

    def _create_socket(self, timeout):
        sock = POP3._create_socket(self, timeout)
        sock = self.context.wrap_socket(sock, server_hostname=self.host)
        return sock

    def stls(self, keyfile=None, certfile=None, context=None):
        """The method unconditionally raises an exception since the
            STLS command doesn't make any sense on an already established
            SSL/TLS session.
            """
        raise error_proto('-ERR TLS session already established')