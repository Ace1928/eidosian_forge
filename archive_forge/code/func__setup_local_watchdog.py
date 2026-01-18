import json
import os
import shutil
import signal
import socket
from string import Template
import tempfile
import uuid
from typing import Any, Dict, Optional, Tuple
import torch.distributed.elastic.timer as timer
from torch.distributed.elastic import events
from torch.distributed.elastic.agent.server.api import (
from torch.distributed.elastic.events.api import EventMetadataValue
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import PContext, start_processes
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
def _setup_local_watchdog(self, envs: Dict[int, Dict[str, str]]) -> None:
    enable_watchdog_env_name = TORCHELASTIC_ENABLE_FILE_TIMER
    watchdog_enabled = os.getenv(enable_watchdog_env_name)
    watchdog_file_env_name = TORCHELASTIC_TIMER_FILE
    watchdog_file_path = os.getenv(watchdog_file_env_name)
    if watchdog_enabled is not None and str(watchdog_enabled) == '1':
        if watchdog_file_path is None:
            watchdog_file_path = '/tmp/watchdog_timer_' + str(uuid.uuid4())
        log.info('Starting a FileTimerServer with %s ...', watchdog_file_path)
        self._worker_watchdog = timer.FileTimerServer(file_path=watchdog_file_path, max_interval=0.1, daemon=True, log_event=self._log_watchdog_event)
        self._worker_watchdog.start()
        log.info('FileTimerServer started')
    else:
        log.info("Environment variable '%s' not found. Do not start FileTimerServer.", enable_watchdog_env_name)
    if watchdog_file_path is not None:
        for worker_env in envs.values():
            worker_env[watchdog_file_env_name] = watchdog_file_path