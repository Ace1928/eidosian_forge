import io
import logging
import os.path
import urllib.parse
from smart_open import bytebuffer, constants
import smart_open.utils
class BufferedInputBase(io.BufferedIOBase):

    def __init__(self, url, mode='r', buffer_size=DEFAULT_BUFFER_SIZE, kerberos=False, user=None, password=None, cert=None, headers=None, timeout=None):
        if kerberos:
            import requests_kerberos
            auth = requests_kerberos.HTTPKerberosAuth()
        elif user is not None and password is not None:
            auth = (user, password)
        else:
            auth = None
        self.buffer_size = buffer_size
        self.mode = mode
        if headers is None:
            self.headers = _HEADERS.copy()
        else:
            self.headers = headers
        self.timeout = timeout
        self.response = requests.get(url, auth=auth, cert=cert, stream=True, headers=self.headers, timeout=self.timeout)
        if not self.response.ok:
            self.response.raise_for_status()
        self._read_iter = self.response.iter_content(self.buffer_size)
        self._read_buffer = bytebuffer.ByteBuffer(buffer_size)
        self._current_pos = 0
        self.raw = None

    def close(self):
        """Flush and close this stream."""
        logger.debug('close: called')
        self.response = None
        self._read_iter = None

    def readable(self):
        """Return True if the stream can be read from."""
        return True

    def seekable(self):
        return False

    def detach(self):
        """Unsupported."""
        raise io.UnsupportedOperation

    def read(self, size=-1):
        """
        Mimics the read call to a filehandle object.
        """
        logger.debug('reading with size: %d', size)
        if self.response is None:
            return b''
        if size == 0:
            return b''
        elif size < 0 and len(self._read_buffer) == 0:
            retval = self.response.raw.read()
        elif size < 0:
            retval = self._read_buffer.read() + self.response.raw.read()
        else:
            while len(self._read_buffer) < size:
                logger.debug('http reading more content at current_pos: %d with size: %d', self._current_pos, size)
                bytes_read = self._read_buffer.fill(self._read_iter)
                if bytes_read == 0:
                    retval = self._read_buffer.read()
                    self._current_pos += len(retval)
                    return retval
            retval = self._read_buffer.read(size)
        self._current_pos += len(retval)
        return retval

    def read1(self, size=-1):
        """This is the same as read()."""
        return self.read(size=size)

    def readinto(self, b):
        """Read up to len(b) bytes into b, and return the number of bytes
        read."""
        data = self.read(len(b))
        if not data:
            return 0
        b[:len(data)] = data
        return len(data)