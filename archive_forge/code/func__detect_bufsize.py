import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def _detect_bufsize(self) -> None:
    sndbuf_size = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    rcvbuf_size = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    self._bufsize = min(sndbuf_size, rcvbuf_size, 65536)