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
def _recv_length(self, bits):
    """Parse the length from the message."""
    length_bits = bits & 127
    if length_bits == 126:
        v = self.recv(2)
        return struct.unpack('!H', v)[0]
    elif length_bits == 127:
        v = self.recv(8)
        return struct.unpack('!Q', v)[0]
    else:
        return length_bits