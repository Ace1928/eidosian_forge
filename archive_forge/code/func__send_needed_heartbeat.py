import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.shared_utils import print_and_log
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def _send_needed_heartbeat(self, connection_id):
    """
        Sends a heartbeat to a connection if needed.
        """
    if connection_id not in self.last_received_heartbeat:
        return
    if self.last_received_heartbeat[connection_id] is None:
        return
    if time.time() - self.last_sent_heartbeat_time[connection_id] < self.HEARTBEAT_RATE:
        return
    packet = self.last_received_heartbeat[connection_id]
    self._safe_send(json.dumps({'type': data_model.SOCKET_ROUTE_PACKET_STRING, 'content': packet.new_copy().swap_sender().set_data('').as_dict()}))
    self.last_sent_heartbeat_time[connection_id] = time.time()