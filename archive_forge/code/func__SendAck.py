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
def _SendAck(self):
    """Send an ACK back to server."""
    if self._total_bytes_received > self._total_bytes_received_and_acked:
        bytes_received = self._total_bytes_received
        try:
            ack_data = utils.CreateSubprotocolAckFrame(bytes_received)
            self._websocket_helper.Send(ack_data)
            self._total_bytes_received_and_acked = bytes_received
        except helper.WebSocketConnectionClosed:
            raise
        except EnvironmentError as e:
            log.info('[%d] Unable to send WebSocket ack [%s]', self._conn_id, six.text_type(e))
        except:
            if not self._IsClosed():
                log.info('[%d] Error while attempting to ack [%d] bytes', self._conn_id, bytes_received, exc_info=True)
            else:
                raise
        finally:
            self._cant_send_ack.clear()