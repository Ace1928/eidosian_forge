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
def _reaper_thread(*args):
    start_time = time.time()
    wait_time = self.DEF_MISSED_PONGS * self.PING_RATE
    while time.time() - start_time < wait_time:
        if self.is_shutdown:
            return
        if self.alive:
            return
        time.sleep(0.3)
    if self.server_death_callback is not None:
        shared_utils.print_and_log(logging.WARN, 'Server has disconnected and could not reconnect. Assuming the worst and calling the death callback. (Usually shutdown)', should_print=True)
        self.server_death_callback()