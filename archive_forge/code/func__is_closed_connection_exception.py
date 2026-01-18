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
def _is_closed_connection_exception(self, exception):
    """Method to identify if the exception is of closed connection type."""
    if exception is websocket_exceptions.WebSocketConnectionClosedException:
        return True
    elif exception is OSError and exception.errno == errno.EBADF:
        return True
    elif exception is ssl.SSLError:
        if exception.args[0] == ssl.SSL_ERROR_EOF:
            return True
    else:
        error_code = websocket_utils.extract_error_code(exception)
        if error_code == errno.ENOTCONN or error_code == errno.EPIPE:
            return True
    return False