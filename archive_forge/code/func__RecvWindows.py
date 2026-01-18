from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ctypes
import errno
import functools
import gc
import io
import os
import select
import socket
import sys
import threading
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.api_lib.compute import sg_tunnel
from googlecloudsdk.api_lib.compute import sg_tunnel_utils as sg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
from six.moves import queue
def _RecvWindows(self, bufsize):
    """Reads data from stdin on Windows.

    Args:
      bufsize: The maximum number of bytes to receive. Must be positive.
    Returns:
      The bytes received. EOF is indicated by b''.
    Raises:
      socket.error: On low level errors.
    """
    if bufsize != utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE:
        log.info('bufsize [%s] is not max_data_frame_size', bufsize)
    while not self._stdin_closed:
        try:
            msg = self._message_queue.get(timeout=1)
        except queue.Empty:
            continue
        msg_type = msg.GetType()
        msg_data = msg.GetData()
        if msg_type is self._ExceptionMessageType:
            six.reraise(msg_data[0], msg_data[1], msg_data[2])
        if msg_type is self._StdinClosedMessageType:
            self._stdin_closed = True
        return msg_data
    return b''