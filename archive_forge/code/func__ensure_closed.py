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
def _ensure_closed(self):
    self.alive = False
    if self.ws is None:
        return
    try:
        self.ws.close()
    except websocket.WebSocketConnectionClosedException:
        pass
    self.ws = None