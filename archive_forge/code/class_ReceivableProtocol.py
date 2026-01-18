from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
class ReceivableProtocol(Protocol):
    """Variant of Protocol that allows reading up to a size without blocking.

    This class has a recv() method that behaves like socket.recv() in addition
    to a read() method.

    If you want to read n bytes from the wire and block until exactly n bytes
    (or EOF) are read, use read(n). If you want to read at most n bytes from
    the wire but don't care if you get less, use recv(n). Note that recv(n)
    will still block until at least one byte is read.
    """

    def __init__(self, recv, write, close=None, report_activity=None, rbufsize=_RBUFSIZE) -> None:
        super().__init__(self.read, write, close=close, report_activity=report_activity)
        self._recv = recv
        self._rbuf = BytesIO()
        self._rbufsize = rbufsize

    def read(self, size):
        assert size > 0
        buf = self._rbuf
        start = buf.tell()
        buf.seek(0, SEEK_END)
        buf_len = buf.tell() - start
        if buf_len >= size:
            buf.seek(start)
            rv = buf.read(size)
            self._rbuf = BytesIO()
            self._rbuf.write(buf.read())
            self._rbuf.seek(0)
            return rv
        self._rbuf = BytesIO()
        while True:
            left = size - buf_len
            data = self._recv(left)
            if not data:
                break
            n = len(data)
            if n == size and (not buf_len):
                return data
            if n == left:
                buf.write(data)
                del data
                break
            assert n <= left, '_recv(%d) returned %d bytes' % (left, n)
            buf.write(data)
            buf_len += n
            del data
        buf.seek(start)
        return buf.read()

    def recv(self, size):
        assert size > 0
        buf = self._rbuf
        start = buf.tell()
        buf.seek(0, SEEK_END)
        buf_len = buf.tell()
        buf.seek(start)
        left = buf_len - start
        if not left:
            data = self._recv(self._rbufsize)
            if len(data) == size:
                return data
            buf = BytesIO()
            buf.write(data)
            buf.seek(0)
            del data
            self._rbuf = buf
        return buf.read(size)