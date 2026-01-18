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
def _SendDataAndReconnectWebSocket(self):
    """Main function for send_and_reconnect_thread."""

    def SendData():
        if not self._stopping:
            self._SendQueuedData()
            self._SendAck()

    def Reconnect():
        if not self._stopping:
            self._StartNewWebSocket()
            self._WaitForOpenOrRaiseError()
    try:
        while not self._stopping:
            try:
                SendData()
            except Exception as e:
                log.debug('[%d] Error while sending data, trying to reconnect [%s]', self._conn_id, six.text_type(e))
                self._AttemptReconnect(Reconnect)
    finally:
        self.Close()