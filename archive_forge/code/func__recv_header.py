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
def _recv_header(self):
    """Parse the header from the message."""
    header = self.recv(2)
    b1 = header[0]
    if six.PY2:
        b1 = ord(b1)
    fin = b1 >> 7 & 1
    rsv1 = b1 >> 6 & 1
    rsv2 = b1 >> 5 & 1
    rsv3 = b1 >> 4 & 1
    opcode = b1 & 15
    b2 = header[1]
    if six.PY2:
        b2 = ord(b2)
    has_mask = b2 >> 7 & 1
    length_bits = b2 & 127
    return (fin, rsv1, rsv2, rsv3, opcode, has_mask, length_bits)