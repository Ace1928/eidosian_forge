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
def _reset_time_logs(self, init_load=False, force=False):
    if not self._should_use_time_logs():
        return
    file_path = os.path.join(parent_dir, TIME_LOGS_FILE_NAME)
    file_lock = os.path.join(parent_dir, TIME_LOGS_FILE_LOCK)
    with LockFile(file_lock) as _lock_file:
        assert _lock_file is not None
        if os.path.exists(file_path):
            with open(file_path, 'rb+') as time_log_file:
                existing_times = pickle.load(time_log_file)
                compare_time = 24 * 60 * 60 if init_load else 60 * 60
                if time.time() - existing_times['last_reset'] < compare_time and (not force):
                    return
                reset_workers = list(existing_times.keys())
                reset_workers.remove('last_reset')
                if len(reset_workers) != 0:
                    self.worker_manager.un_time_block_workers(reset_workers)
            os.remove(file_path)
        with open(file_path, 'wb+') as time_log_file:
            time_logs = {'last_reset': time.time()}
            pickle.dump(time_logs, time_log_file, pickle.HIGHEST_PROTOCOL)