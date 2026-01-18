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
def _log_working_time(self, mturk_agent):
    if not self._should_use_time_logs():
        return
    additional_time = time.time() - mturk_agent.creation_time
    worker_id = mturk_agent.worker_id
    file_path = os.path.join(parent_dir, TIME_LOGS_FILE_NAME)
    file_lock = os.path.join(parent_dir, TIME_LOGS_FILE_LOCK)
    with LockFile(file_lock) as _lock_file:
        assert _lock_file is not None
        if not os.path.exists(file_path):
            self._reset_time_logs()
        with open(file_path, 'rb+') as time_log_file:
            existing_times = pickle.load(time_log_file)
            total_work_time = existing_times.get(worker_id, 0)
            total_work_time += additional_time
            existing_times[worker_id] = total_work_time
        os.remove(file_path)
        with open(file_path, 'wb+') as time_log_file:
            pickle.dump(existing_times, time_log_file, pickle.HIGHEST_PROTOCOL)
    if total_work_time > int(self.opt.get('max_time')):
        self.worker_manager.time_block_worker(worker_id)