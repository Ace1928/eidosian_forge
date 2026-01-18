import sys
import os
from collections import namedtuple
from enum import Enum as _Enum, IntEnum as _IntEnum, IntFlag as _IntFlag
from enum import _simple_enum
import _ssl             # if we can't import it, let the error propagate
from _ssl import OPENSSL_VERSION_NUMBER, OPENSSL_VERSION_INFO, OPENSSL_VERSION
from _ssl import _SSLContext, MemoryBIO, SSLSession
from _ssl import (
from _ssl import txt2obj as _txt2obj, nid2obj as _nid2obj
from _ssl import RAND_status, RAND_add, RAND_bytes, RAND_pseudo_bytes
from _ssl import (
from _ssl import _DEFAULT_CIPHERS, _OPENSSL_API_VERSION
from socket import socket, SOCK_STREAM, create_connection
from socket import SOL_SOCKET, SO_TYPE, _GLOBAL_DEFAULT_TIMEOUT
import socket as _socket
import base64        # for DER-to-PEM translation
import errno
import warnings
class SSLSocket(socket):
    """This class implements a subtype of socket.socket that wraps
    the underlying OS socket in an SSL context when necessary, and
    provides read and write methods over that channel. """

    def __init__(self, *args, **kwargs):
        raise TypeError(f'{self.__class__.__name__} does not have a public constructor. Instances are returned by SSLContext.wrap_socket().')

    @classmethod
    def _create(cls, sock, server_side=False, do_handshake_on_connect=True, suppress_ragged_eofs=True, server_hostname=None, context=None, session=None):
        if sock.getsockopt(SOL_SOCKET, SO_TYPE) != SOCK_STREAM:
            raise NotImplementedError('only stream sockets are supported')
        if server_side:
            if server_hostname:
                raise ValueError('server_hostname can only be specified in client mode')
            if session is not None:
                raise ValueError('session can only be specified in client mode')
        if context.check_hostname and (not server_hostname):
            raise ValueError('check_hostname requires server_hostname')
        sock_timeout = sock.gettimeout()
        kwargs = dict(family=sock.family, type=sock.type, proto=sock.proto, fileno=sock.fileno())
        self = cls.__new__(cls, **kwargs)
        super(SSLSocket, self).__init__(**kwargs)
        sock.detach()
        try:
            self._context = context
            self._session = session
            self._closed = False
            self._sslobj = None
            self.server_side = server_side
            self.server_hostname = context._encode_hostname(server_hostname)
            self.do_handshake_on_connect = do_handshake_on_connect
            self.suppress_ragged_eofs = suppress_ragged_eofs
            try:
                self.getpeername()
            except OSError as e:
                if e.errno != errno.ENOTCONN:
                    raise
                connected = False
                blocking = self.getblocking()
                self.setblocking(False)
                try:
                    notconn_pre_handshake_data = self.recv(1)
                except OSError as e:
                    if e.errno not in (errno.ENOTCONN, errno.EINVAL):
                        raise
                    notconn_pre_handshake_data = b''
                self.setblocking(blocking)
                if notconn_pre_handshake_data:
                    reason = 'Closed before TLS handshake with data in recv buffer.'
                    notconn_pre_handshake_data_error = SSLError(e.errno, reason)
                    notconn_pre_handshake_data_error.reason = reason
                    notconn_pre_handshake_data_error.library = None
                    try:
                        raise notconn_pre_handshake_data_error
                    finally:
                        notconn_pre_handshake_data_error = None
            else:
                connected = True
            self.settimeout(sock_timeout)
            self._connected = connected
            if connected:
                self._sslobj = self._context._wrap_socket(self, server_side, self.server_hostname, owner=self, session=self._session)
                if do_handshake_on_connect:
                    timeout = self.gettimeout()
                    if timeout == 0.0:
                        raise ValueError('do_handshake_on_connect should not be specified for non-blocking sockets')
                    self.do_handshake()
        except:
            try:
                self.close()
            except OSError:
                pass
            raise
        return self

    @property
    @_sslcopydoc
    def context(self):
        return self._context

    @context.setter
    def context(self, ctx):
        self._context = ctx
        self._sslobj.context = ctx

    @property
    @_sslcopydoc
    def session(self):
        if self._sslobj is not None:
            return self._sslobj.session

    @session.setter
    def session(self, session):
        self._session = session
        if self._sslobj is not None:
            self._sslobj.session = session

    @property
    @_sslcopydoc
    def session_reused(self):
        if self._sslobj is not None:
            return self._sslobj.session_reused

    def dup(self):
        raise NotImplementedError("Can't dup() %s instances" % self.__class__.__name__)

    def _checkClosed(self, msg=None):
        pass

    def _check_connected(self):
        if not self._connected:
            self.getpeername()

    def read(self, len=1024, buffer=None):
        """Read up to LEN bytes and return them.
        Return zero-length string on EOF."""
        self._checkClosed()
        if self._sslobj is None:
            raise ValueError('Read on closed or unwrapped SSL socket.')
        try:
            if buffer is not None:
                return self._sslobj.read(len, buffer)
            else:
                return self._sslobj.read(len)
        except SSLError as x:
            if x.args[0] == SSL_ERROR_EOF and self.suppress_ragged_eofs:
                if buffer is not None:
                    return 0
                else:
                    return b''
            else:
                raise

    def write(self, data):
        """Write DATA to the underlying SSL channel.  Returns
        number of bytes of DATA actually transmitted."""
        self._checkClosed()
        if self._sslobj is None:
            raise ValueError('Write on closed or unwrapped SSL socket.')
        return self._sslobj.write(data)

    @_sslcopydoc
    def getpeercert(self, binary_form=False):
        self._checkClosed()
        self._check_connected()
        return self._sslobj.getpeercert(binary_form)

    @_sslcopydoc
    def selected_npn_protocol(self):
        self._checkClosed()
        warnings.warn('ssl NPN is deprecated, use ALPN instead', DeprecationWarning, stacklevel=2)
        return None

    @_sslcopydoc
    def selected_alpn_protocol(self):
        self._checkClosed()
        if self._sslobj is None or not _ssl.HAS_ALPN:
            return None
        else:
            return self._sslobj.selected_alpn_protocol()

    @_sslcopydoc
    def cipher(self):
        self._checkClosed()
        if self._sslobj is None:
            return None
        else:
            return self._sslobj.cipher()

    @_sslcopydoc
    def shared_ciphers(self):
        self._checkClosed()
        if self._sslobj is None:
            return None
        else:
            return self._sslobj.shared_ciphers()

    @_sslcopydoc
    def compression(self):
        self._checkClosed()
        if self._sslobj is None:
            return None
        else:
            return self._sslobj.compression()

    def send(self, data, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError('non-zero flags not allowed in calls to send() on %s' % self.__class__)
            return self._sslobj.write(data)
        else:
            return super().send(data, flags)

    def sendto(self, data, flags_or_addr, addr=None):
        self._checkClosed()
        if self._sslobj is not None:
            raise ValueError('sendto not allowed on instances of %s' % self.__class__)
        elif addr is None:
            return super().sendto(data, flags_or_addr)
        else:
            return super().sendto(data, flags_or_addr, addr)

    def sendmsg(self, *args, **kwargs):
        raise NotImplementedError('sendmsg not allowed on instances of %s' % self.__class__)

    def sendall(self, data, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError('non-zero flags not allowed in calls to sendall() on %s' % self.__class__)
            count = 0
            with memoryview(data) as view, view.cast('B') as byte_view:
                amount = len(byte_view)
                while count < amount:
                    v = self.send(byte_view[count:])
                    count += v
        else:
            return super().sendall(data, flags)

    def sendfile(self, file, offset=0, count=None):
        """Send a file, possibly by using os.sendfile() if this is a
        clear-text socket.  Return the total number of bytes sent.
        """
        if self._sslobj is not None:
            return self._sendfile_use_send(file, offset, count)
        else:
            return super().sendfile(file, offset, count)

    def recv(self, buflen=1024, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError('non-zero flags not allowed in calls to recv() on %s' % self.__class__)
            return self.read(buflen)
        else:
            return super().recv(buflen, flags)

    def recv_into(self, buffer, nbytes=None, flags=0):
        self._checkClosed()
        if nbytes is None:
            if buffer is not None:
                with memoryview(buffer) as view:
                    nbytes = view.nbytes
                if not nbytes:
                    nbytes = 1024
            else:
                nbytes = 1024
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError('non-zero flags not allowed in calls to recv_into() on %s' % self.__class__)
            return self.read(nbytes, buffer)
        else:
            return super().recv_into(buffer, nbytes, flags)

    def recvfrom(self, buflen=1024, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            raise ValueError('recvfrom not allowed on instances of %s' % self.__class__)
        else:
            return super().recvfrom(buflen, flags)

    def recvfrom_into(self, buffer, nbytes=None, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            raise ValueError('recvfrom_into not allowed on instances of %s' % self.__class__)
        else:
            return super().recvfrom_into(buffer, nbytes, flags)

    def recvmsg(self, *args, **kwargs):
        raise NotImplementedError('recvmsg not allowed on instances of %s' % self.__class__)

    def recvmsg_into(self, *args, **kwargs):
        raise NotImplementedError('recvmsg_into not allowed on instances of %s' % self.__class__)

    @_sslcopydoc
    def pending(self):
        self._checkClosed()
        if self._sslobj is not None:
            return self._sslobj.pending()
        else:
            return 0

    def shutdown(self, how):
        self._checkClosed()
        self._sslobj = None
        super().shutdown(how)

    @_sslcopydoc
    def unwrap(self):
        if self._sslobj:
            s = self._sslobj.shutdown()
            self._sslobj = None
            return s
        else:
            raise ValueError('No SSL wrapper around ' + str(self))

    @_sslcopydoc
    def verify_client_post_handshake(self):
        if self._sslobj:
            return self._sslobj.verify_client_post_handshake()
        else:
            raise ValueError('No SSL wrapper around ' + str(self))

    def _real_close(self):
        self._sslobj = None
        super()._real_close()

    @_sslcopydoc
    def do_handshake(self, block=False):
        self._check_connected()
        timeout = self.gettimeout()
        try:
            if timeout == 0.0 and block:
                self.settimeout(None)
            self._sslobj.do_handshake()
        finally:
            self.settimeout(timeout)

    def _real_connect(self, addr, connect_ex):
        if self.server_side:
            raise ValueError("can't connect in server-side mode")
        if self._connected or self._sslobj is not None:
            raise ValueError('attempt to connect already-connected SSLSocket!')
        self._sslobj = self.context._wrap_socket(self, False, self.server_hostname, owner=self, session=self._session)
        try:
            if connect_ex:
                rc = super().connect_ex(addr)
            else:
                rc = None
                super().connect(addr)
            if not rc:
                self._connected = True
                if self.do_handshake_on_connect:
                    self.do_handshake()
            return rc
        except (OSError, ValueError):
            self._sslobj = None
            raise

    def connect(self, addr):
        """Connects to remote ADDR, and then wraps the connection in
        an SSL channel."""
        self._real_connect(addr, False)

    def connect_ex(self, addr):
        """Connects to remote ADDR, and then wraps the connection in
        an SSL channel."""
        return self._real_connect(addr, True)

    def accept(self):
        """Accepts a new connection from a remote client, and returns
        a tuple containing that new connection wrapped with a server-side
        SSL channel, and the address of the remote client."""
        newsock, addr = super().accept()
        newsock = self.context.wrap_socket(newsock, do_handshake_on_connect=self.do_handshake_on_connect, suppress_ragged_eofs=self.suppress_ragged_eofs, server_side=True)
        return (newsock, addr)

    @_sslcopydoc
    def get_channel_binding(self, cb_type='tls-unique'):
        if self._sslobj is not None:
            return self._sslobj.get_channel_binding(cb_type)
        else:
            if cb_type not in CHANNEL_BINDING_TYPES:
                raise ValueError('{0} channel binding type not implemented'.format(cb_type))
            return None

    @_sslcopydoc
    def version(self):
        if self._sslobj is not None:
            return self._sslobj.version()
        else:
            return None