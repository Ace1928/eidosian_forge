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
def _CleanDeadClientConnections(self):
    """Erase reference to dead connections so they can be garbage collected."""
    conn_still_alive = []
    if self._connections:
        dead_connections = 0
        for client_thread, conn in self._connections:
            if not client_thread.is_alive():
                dead_connections += 1
                try:
                    conn.close()
                except EnvironmentError:
                    pass
                del conn
                del client_thread
            else:
                conn_still_alive.append([client_thread, conn])
        if dead_connections:
            log.debug('Cleaned [%d] dead connection(s).' % dead_connections)
            self._connections = conn_still_alive
        gc.collect(2)
        log.debug('connections alive: [%d]' % len(self._connections))