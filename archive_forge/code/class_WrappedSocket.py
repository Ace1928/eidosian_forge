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
class WrappedSocket(object):
    """API-compatibility wrapper for Python OpenSSL's Connection-class.

    Note: _makefile_refs, _drop() and _reuse() are needed for the garbage
    collector of pypy.
    """

    def __init__(self, connection, socket, suppress_ragged_eofs=True):
        self.connection = connection
        self.socket = socket
        self.suppress_ragged_eofs = suppress_ragged_eofs
        self._makefile_refs = 0
        self._closed = False

    def fileno(self):
        return self.socket.fileno()

    def _decref_socketios(self):
        if self._makefile_refs > 0:
            self._makefile_refs -= 1
        if self._closed:
            self.close()

    def recv(self, *args, **kwargs):
        try:
            data = self.connection.recv(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
                return b''
            else:
                raise SocketError(str(e))
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return b''
            else:
                raise
        except OpenSSL.SSL.WantReadError:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise timeout('The read operation timed out')
            else:
                return self.recv(*args, **kwargs)
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError('read error: %r' % e)
        else:
            return data

    def recv_into(self, *args, **kwargs):
        try:
            return self.connection.recv_into(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
                return 0
            else:
                raise SocketError(str(e))
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return 0
            else:
                raise
        except OpenSSL.SSL.WantReadError:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise timeout('The read operation timed out')
            else:
                return self.recv_into(*args, **kwargs)
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError('read error: %r' % e)

    def settimeout(self, timeout):
        return self.socket.settimeout(timeout)

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

    def sendall(self, data):
        total_sent = 0
        while total_sent < len(data):
            sent = self._send_until_done(data[total_sent:total_sent + SSL_WRITE_BLOCKSIZE])
            total_sent += sent

    def shutdown(self):
        self.connection.shutdown()

    def close(self):
        if self._makefile_refs < 1:
            try:
                self._closed = True
                return self.connection.close()
            except OpenSSL.SSL.Error:
                return
        else:
            self._makefile_refs -= 1

    def getpeercert(self, binary_form=False):
        x509 = self.connection.get_peer_certificate()
        if not x509:
            return x509
        if binary_form:
            return OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_ASN1, x509)
        return {'subject': ((('commonName', x509.get_subject().CN),),), 'subjectAltName': get_subj_alt_name(x509)}

    def version(self):
        return self.connection.get_protocol_version_name()

    def _reuse(self):
        self._makefile_refs += 1

    def _drop(self):
        if self._makefile_refs < 1:
            self.close()
        else:
            self._makefile_refs -= 1