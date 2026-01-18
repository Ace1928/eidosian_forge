import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.dev.shared_utils import print_and_log
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def _try_send_world_ping(self):
    if time.time() - self.last_sent_ping_time > self.PING_RATE:
        self._safe_send(json.dumps({'type': data_model.WORLD_PING, 'content': {'id': 'WORLD_PING', 'sender_id': self.get_my_sender_id()}}), force=True)
        self.last_sent_ping_time = time.time()