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
def _get_close_args(self, data):
    if data and len(data) >= 2:
        code = 256 * six.byte2int(data[0:1]) + six.byte2int(data[1:2])
        reason = data[2:].decode('utf-8')
        return [code, reason]