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
def _safe_send(self, data, force=False):
    if not self.alive and (not force):
        timeout = 0.5
        while timeout > 0 and (not self.alive):
            time.sleep(0.1)
            timeout -= 0.1
        if not self.alive:
            return False
    try:
        with self.send_lock:
            self.ws.send(data)
    except websocket.WebSocketConnectionClosedException:
        return False
    except BrokenPipeError:
        return False
    except AttributeError:
        return False
    except Exception as e:
        shared_utils.print_and_log(logging.WARN, 'Unexpected socket error occured: {}'.format(repr(e)), should_print=True)
        return False
    return True