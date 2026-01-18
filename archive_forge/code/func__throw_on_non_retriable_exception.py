from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import select
import socket
import ssl
import struct
import threading
from googlecloudsdk.core.util import platforms
import six
import websocket._abnf as websocket_frame_utils
import websocket._exceptions as websocket_exceptions
import websocket._handshake as websocket_handshake
import websocket._http as websocket_http_utils
import websocket._utils as websocket_utils
def _throw_on_non_retriable_exception(self, e):
    """Decides if we throw or if we ignore the exception because it's retriable."""
    if self._is_closed_connection_exception(e):
        raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while waiting for retry.')
    if e is ssl.SSLError:
        if e.args[0] != ssl.SSL_ERROR_WANT_WRITE:
            raise e
    elif e is socket.error:
        error_code = websocket_utils.extract_error_code(e)
        if error_code is None:
            raise e
        if error_code != errno.EAGAIN or error_code != errno.EWOULDBLOCK:
            raise e