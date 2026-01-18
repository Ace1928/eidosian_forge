from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import logging
import threading
import time
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_helper as helper
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
import six
from six.moves import queue
def _HandleSubprotocolData(self, binary_data):
    """Handle Subprotocol DATA Frame."""
    if not self._HasConnected():
        self._StopConnectionAsync()
        raise SubprotocolEarlyDataError('Received DATA before connected.')
    data, bytes_left = utils.ExtractSubprotocolData(binary_data)
    self._total_bytes_received += len(data)
    self._MaybeSendAck()
    try:
        self._data_handler_callback(data)
    except:
        self._StopConnectionAsync()
        raise
    if bytes_left:
        log.debug('[%d] Discarding [%d] extra bytes after processing DATA', self._conn_id, len(bytes_left))