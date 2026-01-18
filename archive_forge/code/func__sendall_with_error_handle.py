import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def _sendall_with_error_handle(self, data: bytes) -> None:
    total_sent = 0
    total_data = len(data)
    while total_sent < total_data:
        start_time = time.monotonic()
        try:
            sent = self._sock.send(data)
            if sent == 0:
                raise SockClientClosedError('socket connection broken')
            total_sent += sent
            data = data[sent:]
        except socket.timeout:
            delta_time = time.monotonic() - start_time
            if delta_time < self._retry_delay:
                time.sleep(self._retry_delay - delta_time)