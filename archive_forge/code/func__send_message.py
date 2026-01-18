import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def _send_message(self, msg: Any) -> None:
    tracelog.log_message_send(msg, self._sockid)
    raw_size = msg.ByteSize()
    data = msg.SerializeToString()
    assert len(data) == raw_size, 'invalid serialization'
    header = struct.pack('<BI', ord('W'), raw_size)
    with self._lock:
        self._sendall_with_error_handle(header + data)