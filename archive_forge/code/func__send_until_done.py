from __future__ import absolute_import
import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.hazmat.backends.openssl.x509 import _Certificate
from io import BytesIO
from socket import error as SocketError
from socket import timeout
import logging
import ssl
import sys
from .. import util
from ..packages import six
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
def _send_until_done(self, data):
    while True:
        try:
            return self.connection.send(data)
        except OpenSSL.SSL.WantWriteError:
            if not util.wait_for_write(self.socket, self.socket.gettimeout()):
                raise timeout()
            continue
        except OpenSSL.SSL.SysCallError as e:
            raise SocketError(str(e))