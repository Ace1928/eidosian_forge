import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def _check_time_limit(self):
    if time.time() - self.time_limit_checked < RESET_TIME_LOG_TIMEOUT:
        return
    if int(time.time()) % (60 * 60 * 24) > 60 * 30:
        return
    self.time_limit_checked = time.time()
    self._reset_time_logs()
    self.worker_manager.un_time_block_workers()