import socket
from contextlib import contextmanager
from .exceptions import TIMEOUT, EOF
from .spawnbase import SpawnBase
class SocketSpawn(SpawnBase):
    """This is like :mod:`pexpect.fdpexpect` but uses the cross-platform python socket api,
    rather than the unix-specific file descriptor api. Thus, it works with
    remote connections on both unix and windows."""

    def __init__(self, socket: socket.socket, args=None, timeout=30, maxread=2000, searchwindowsize=None, logfile=None, encoding=None, codec_errors='strict', use_poll=False):
        """This takes an open socket."""
        self.args = None
        self.command = None
        SpawnBase.__init__(self, timeout, maxread, searchwindowsize, logfile, encoding=encoding, codec_errors=codec_errors)
        self.socket = socket
        self.child_fd = socket.fileno()
        self.closed = False
        self.name = '<socket %s>' % socket
        self.use_poll = use_poll

    def close(self):
        """Close the socket.

        Calling this method a second time does nothing, but if the file
        descriptor was closed elsewhere, :class:`OSError` will be raised.
        """
        if self.child_fd == -1:
            return
        self.flush()
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.child_fd = -1
        self.closed = True

    def isalive(self):
        """ Alive if the fileno is valid """
        return self.socket.fileno() >= 0

    def send(self, s) -> int:
        """Write to socket, return number of bytes written"""
        s = self._coerce_send_string(s)
        self._log(s, 'send')
        b = self._encoder.encode(s, final=False)
        self.socket.sendall(b)
        return len(b)

    def sendline(self, s) -> int:
        """Write to socket with trailing newline, return number of bytes written"""
        s = self._coerce_send_string(s)
        return self.send(s + self.linesep)

    def write(self, s):
        """Write to socket, return None"""
        self.send(s)

    def writelines(self, sequence):
        """Call self.write() for each item in sequence"""
        for s in sequence:
            self.write(s)

    @contextmanager
    def _timeout(self, timeout):
        saved_timeout = self.socket.gettimeout()
        try:
            self.socket.settimeout(timeout)
            yield
        finally:
            self.socket.settimeout(saved_timeout)

    def read_nonblocking(self, size=1, timeout=-1):
        """
        Read from the file descriptor and return the result as a string.

        The read_nonblocking method of :class:`SpawnBase` assumes that a call
        to os.read will not block (timeout parameter is ignored). This is not
        the case for POSIX file-like objects such as sockets and serial ports.

        Use :func:`select.select`, timeout is implemented conditionally for
        POSIX systems.

        :param int size: Read at most *size* bytes.
        :param int timeout: Wait timeout seconds for file descriptor to be
            ready to read. When -1 (default), use self.timeout. When 0, poll.
        :return: String containing the bytes read
        """
        if timeout == -1:
            timeout = self.timeout
        try:
            with self._timeout(timeout):
                s = self.socket.recv(size)
                if s == b'':
                    self.flag_eof = True
                    raise EOF('Socket closed')
                return s
        except socket.timeout:
            raise TIMEOUT('Timeout exceeded.')