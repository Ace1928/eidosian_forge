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
def _wait_for_task_expirations(self):
    """
        Wait for the full task duration to ensure anyone who sees the task has it
        expired, and ensures that all tasks are properly expired.
        """
    start_time = time.time()
    min_wait = self.opt['assignment_duration_in_seconds']
    wait_time = 0.1
    while time.time() - start_time < min_wait and len(self.hit_id_list) > 0:
        for hit_id in self.hit_id_list.copy():
            self.update_hit_status(hit_id)
        wait_time *= 1.5
        time.sleep(wait_time)