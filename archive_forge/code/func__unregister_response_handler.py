import dataclasses
import socket
import ssl
import threading
import typing as t
def _unregister_response_handler(self, handler: ResponseHandler) -> None:
    self._response_handler.remove(handler)