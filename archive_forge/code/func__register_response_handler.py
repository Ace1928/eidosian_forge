import dataclasses
import socket
import ssl
import threading
import typing as t
def _register_response_handler(self, msg_id: int, *message_types: t.Type[MessageType]) -> ResponseHandler[MessageType]:
    handler = ResponseHandler(msg_id, message_types)
    self._response_handler.append(handler)
    return handler