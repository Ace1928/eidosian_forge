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
def _AcceptNewConnection(self):
    """Accept a new socket connection and start a new WebSocket tunnel."""
    ready_sockets = [()]
    while not ready_sockets[0]:
        ready_sockets = select.select(self._server_sockets, (), (), 0.2)
    ready_read_sockets = ready_sockets[0]
    conn, socket_address = ready_read_sockets[0].accept()
    new_thread = threading.Thread(target=self._HandleNewConnection, args=(conn, socket_address, self._total_connections))
    new_thread.daemon = True
    new_thread.start()
    self._total_connections += 1
    return (new_thread, conn)