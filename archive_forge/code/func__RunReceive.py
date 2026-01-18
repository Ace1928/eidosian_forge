from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import http.client
import logging
import select
import socket
import ssl
import threading
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as iap_utils
from googlecloudsdk.api_lib.compute import sg_tunnel_utils as sg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def _RunReceive(self):
    """Receives server data and sends to the local connection."""
    try:
        while not self._stopping:
            if not self._sock:
                break
            ready = [[self._sock]]
            if not self._sock.pending():
                ready = select.select([self._sock, self._rpair], (), (), RECV_TIMEOUT_SECONDS)
            for s in ready[0]:
                if s is self._rpair:
                    self._stopping = True
                    break
                if s is self._sock:
                    data, data_len = self._Read()
                    if log.GetVerbosity() == logging.DEBUG:
                        log.err.GetConsoleWriterStream().write('DEBUG: RECV data_len [%d] data[:20] %r\n' % (len(data), data[:20]))
                    if data_len == 0:
                        log.debug('Remote endpoint [%s:%d] closed connection', self._target.host, self._target.port)
                        self._send_local_data_callback(b'')
                        self._stopping = True
                        break
                    if data_len > 0:
                        self._send_local_data_callback(data)
    finally:
        self._stopping = True