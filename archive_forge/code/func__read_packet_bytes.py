import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def _read_packet_bytes(self, timeout: Optional[int]=None) -> Optional[bytes]:
    """Read full message from socket.

        Args:
            timeout: number of seconds to wait on socket data.

        Raises:
            SockClientClosedError: socket has been closed.
        """
    while True:
        rec = self._extract_packet_bytes()
        if rec:
            return rec
        if timeout:
            self._sock.settimeout(timeout)
        try:
            data = self._sock.recv(self._bufsize)
        except socket.timeout:
            break
        except ConnectionResetError:
            raise SockClientClosedError
        except OSError:
            raise SockClientClosedError
        finally:
            if timeout:
                self._sock.settimeout(None)
        data_len = len(data)
        if data_len == 0:
            raise SockClientClosedError
        self._buffer.put(data, data_len)
    return None