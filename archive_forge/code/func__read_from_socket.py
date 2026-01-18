import errno
import io
import socket
from io import SEEK_END
from typing import Optional, Union
from ..exceptions import ConnectionError, TimeoutError
from ..utils import SSL_AVAILABLE
def _read_from_socket(self, length: Optional[int]=None, timeout: Union[float, object]=SENTINEL, raise_on_timeout: Optional[bool]=True) -> bool:
    sock = self._sock
    socket_read_size = self.socket_read_size
    marker = 0
    custom_timeout = timeout is not SENTINEL
    buf = self._buffer
    current_pos = buf.tell()
    buf.seek(0, SEEK_END)
    if custom_timeout:
        sock.settimeout(timeout)
    try:
        while True:
            data = self._sock.recv(socket_read_size)
            if isinstance(data, bytes) and len(data) == 0:
                raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
            buf.write(data)
            data_length = len(data)
            marker += data_length
            if length is not None and length > marker:
                continue
            return True
    except socket.timeout:
        if raise_on_timeout:
            raise TimeoutError('Timeout reading from socket')
        return False
    except NONBLOCKING_EXCEPTIONS as ex:
        allowed = NONBLOCKING_EXCEPTION_ERROR_NUMBERS.get(ex.__class__, -1)
        if not raise_on_timeout and ex.errno == allowed:
            return False
        raise ConnectionError(f'Error while reading from socket: {ex.args}')
    finally:
        buf.seek(current_pos)
        if custom_timeout:
            sock.settimeout(self.socket_timeout)