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
def _EnqueueBytesWithWaitForReconnect(self, bytes_to_send):
    """Add bytes to the queue; block waiting for reconnect if queue is full.

    Args:
      bytes_to_send: The local bytes to send over the websocket. At most
        utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE.

    Raises:
      ConnectionReconnectTimeout: If something is preventing data from being
        sent.
      ConnectionCreationError: If the connection was closed and no more
        reconnect retries will be performed.
    """
    end_time = time.time() + MAX_RECONNECT_WAIT_TIME_MS / 1000.0
    while time.time() < end_time and (not self._stopping):
        try:
            self._unsent_data.put(bytes_to_send, timeout=MAX_WEBSOCKET_SEND_WAIT_TIME_SEC)
            if log.GetVerbosity() == logging.DEBUG:
                log.debug('[%d] ENQUEUED data_len [%d] bytes_to_send[:20] [%r]', self._conn_id, len(bytes_to_send), bytes_to_send[:20])
            return
        except queue.Full:
            pass
    if self._stopping:
        raise ConnectionCreationError('Unexpected error while reconnecting. Check logs for more details.')
    raise ConnectionReconnectTimeout()