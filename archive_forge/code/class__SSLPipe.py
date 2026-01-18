import collections
import sys
import warnings
from . import protocols
from . import transports
from .log import logger
class _SSLPipe(object):
    """An SSL "Pipe".

    An SSL pipe allows you to communicate with an SSL/TLS protocol instance
    through memory buffers. It can be used to implement a security layer for an
    existing connection where you don't have access to the connection's file
    descriptor, or for some reason you don't want to use it.

    An SSL pipe can be in "wrapped" and "unwrapped" mode. In unwrapped mode,
    data is passed through untransformed. In wrapped mode, application level
    data is encrypted to SSL record level data and vice versa. The SSL record
    level is the lowest level in the SSL protocol suite and is what travels
    as-is over the wire.

    An SslPipe initially is in "unwrapped" mode. To start SSL, call
    do_handshake(). To shutdown SSL again, call unwrap().
    """
    max_size = 256 * 1024

    def __init__(self, context, server_side, server_hostname=None):
        """
        The *context* argument specifies the ssl.SSLContext to use.

        The *server_side* argument indicates whether this is a server side or
        client side transport.

        The optional *server_hostname* argument can be used to specify the
        hostname you are connecting to. You may only specify this parameter if
        the _ssl module supports Server Name Indication (SNI).
        """
        self._context = context
        self._server_side = server_side
        self._server_hostname = server_hostname
        self._state = _UNWRAPPED
        self._incoming = ssl.MemoryBIO()
        self._outgoing = ssl.MemoryBIO()
        self._sslobj = None
        self._need_ssldata = False
        self._handshake_cb = None
        self._shutdown_cb = None

    @property
    def context(self):
        """The SSL context passed to the constructor."""
        return self._context

    @property
    def ssl_object(self):
        """The internal ssl.SSLObject instance.

        Return None if the pipe is not wrapped.
        """
        return self._sslobj

    @property
    def need_ssldata(self):
        """Whether more record level data is needed to complete a handshake
        that is currently in progress."""
        return self._need_ssldata

    @property
    def wrapped(self):
        """
        Whether a security layer is currently in effect.

        Return False during handshake.
        """
        return self._state == _WRAPPED

    def do_handshake(self, callback=None):
        """Start the SSL handshake.

        Return a list of ssldata. A ssldata element is a list of buffers

        The optional *callback* argument can be used to install a callback that
        will be called when the handshake is complete. The callback will be
        called with None if successful, else an exception instance.
        """
        if self._state != _UNWRAPPED:
            raise RuntimeError('handshake in progress or completed')
        self._sslobj = self._context.wrap_bio(self._incoming, self._outgoing, server_side=self._server_side, server_hostname=self._server_hostname)
        self._state = _DO_HANDSHAKE
        self._handshake_cb = callback
        ssldata, appdata = self.feed_ssldata(b'', only_handshake=True)
        assert len(appdata) == 0
        return ssldata

    def shutdown(self, callback=None):
        """Start the SSL shutdown sequence.

        Return a list of ssldata. A ssldata element is a list of buffers

        The optional *callback* argument can be used to install a callback that
        will be called when the shutdown is complete. The callback will be
        called without arguments.
        """
        if self._state == _UNWRAPPED:
            raise RuntimeError('no security layer present')
        if self._state == _SHUTDOWN:
            raise RuntimeError('shutdown in progress')
        assert self._state in (_WRAPPED, _DO_HANDSHAKE)
        self._state = _SHUTDOWN
        self._shutdown_cb = callback
        ssldata, appdata = self.feed_ssldata(b'')
        assert appdata == [] or appdata == [b'']
        return ssldata

    def feed_eof(self):
        """Send a potentially "ragged" EOF.

        This method will raise an SSL_ERROR_EOF exception if the EOF is
        unexpected.
        """
        self._incoming.write_eof()
        ssldata, appdata = self.feed_ssldata(b'')
        assert appdata == [] or appdata == [b'']

    def feed_ssldata(self, data, only_handshake=False):
        """Feed SSL record level data into the pipe.

        The data must be a bytes instance. It is OK to send an empty bytes
        instance. This can be used to get ssldata for a handshake initiated by
        this endpoint.

        Return a (ssldata, appdata) tuple. The ssldata element is a list of
        buffers containing SSL data that needs to be sent to the remote SSL.

        The appdata element is a list of buffers containing plaintext data that
        needs to be forwarded to the application. The appdata list may contain
        an empty buffer indicating an SSL "close_notify" alert. This alert must
        be acknowledged by calling shutdown().
        """
        if self._state == _UNWRAPPED:
            if data:
                appdata = [data]
            else:
                appdata = []
            return ([], appdata)
        self._need_ssldata = False
        if data:
            self._incoming.write(data)
        ssldata = []
        appdata = []
        try:
            if self._state == _DO_HANDSHAKE:
                self._sslobj.do_handshake()
                self._state = _WRAPPED
                if self._handshake_cb:
                    self._handshake_cb(None)
                if only_handshake:
                    return (ssldata, appdata)
            if self._state == _WRAPPED:
                while True:
                    chunk = self._sslobj.read(self.max_size)
                    appdata.append(chunk)
                    if not chunk:
                        break
            elif self._state == _SHUTDOWN:
                self._sslobj.unwrap()
                self._sslobj = None
                self._state = _UNWRAPPED
                if self._shutdown_cb:
                    self._shutdown_cb()
            elif self._state == _UNWRAPPED:
                appdata.append(self._incoming.read())
        except (ssl.SSLError, ssl.CertificateError) as exc:
            if getattr(exc, 'errno', None) not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE, ssl.SSL_ERROR_SYSCALL):
                if self._state == _DO_HANDSHAKE and self._handshake_cb:
                    self._handshake_cb(exc)
                raise
            self._need_ssldata = exc.errno == ssl.SSL_ERROR_WANT_READ
        if self._outgoing.pending:
            ssldata.append(self._outgoing.read())
        return (ssldata, appdata)

    def feed_appdata(self, data, offset=0):
        """Feed plaintext data into the pipe.

        Return an (ssldata, offset) tuple. The ssldata element is a list of
        buffers containing record level data that needs to be sent to the
        remote SSL instance. The offset is the number of plaintext bytes that
        were processed, which may be less than the length of data.

        NOTE: In case of short writes, this call MUST be retried with the SAME
        buffer passed into the *data* argument (i.e. the id() must be the
        same). This is an OpenSSL requirement. A further particularity is that
        a short write will always have offset == 0, because the _ssl module
        does not enable partial writes. And even though the offset is zero,
        there will still be encrypted data in ssldata.
        """
        assert 0 <= offset <= len(data)
        if self._state == _UNWRAPPED:
            if offset < len(data):
                ssldata = [data[offset:]]
            else:
                ssldata = []
            return (ssldata, len(data))
        ssldata = []
        view = memoryview(data)
        while True:
            self._need_ssldata = False
            try:
                if offset < len(view):
                    offset += self._sslobj.write(view[offset:])
            except ssl.SSLError as exc:
                if exc.reason == 'PROTOCOL_IS_SHUTDOWN':
                    exc.errno = ssl.SSL_ERROR_WANT_READ
                if exc.errno not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE, ssl.SSL_ERROR_SYSCALL):
                    raise
                self._need_ssldata = exc.errno == ssl.SSL_ERROR_WANT_READ
            if self._outgoing.pending:
                ssldata.append(self._outgoing.read())
            if offset == len(view) or self._need_ssldata:
                break
        return (ssldata, offset)