import dataclasses
import socket
import ssl
import threading
import typing as t
def _write_and_wait_one(self, msg_id: int, message_type: t.Type[MessageType]) -> MessageType:
    handler = self._register_response_handler(msg_id, message_type)
    try:
        self._write_msg()
        return handler.__iter__().__next__()
    finally:
        self._unregister_response_handler(handler)