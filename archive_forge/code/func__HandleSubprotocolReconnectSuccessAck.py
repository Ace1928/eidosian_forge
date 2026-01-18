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
def _HandleSubprotocolReconnectSuccessAck(self, binary_data):
    """Handle Subprotocol RECONNECT_SUCCESS_ACK Frame."""
    if self._HasConnected():
        self._StopConnectionAsync()
        raise SubprotocolExtraReconnectSuccessAck('Received RECONNECT_SUCCESS_ACK after already connected.')
    bytes_confirmed, bytes_left = utils.ExtractSubprotocolReconnectSuccessAck(binary_data)
    bytes_being_confirmed = bytes_confirmed - self._total_bytes_confirmed
    self._ConfirmData(bytes_confirmed)
    log.info('[%d] Reconnecting: confirming [%d] bytes and resending [%d] messages.', self._conn_id, bytes_being_confirmed, len(self._unconfirmed_data))
    self._AddUnconfirmedDataBackToTheQueue()
    self._connect_msg_received = True
    if bytes_left:
        log.debug('[%d] Discarding [%d] extra bytes after processing RECONNECT_SUCCESS_ACK', self._conn_id, len(bytes_left))