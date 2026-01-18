import asyncio
import copy
import json
import logging
import os
import psutil
import random
import signal
import string
import subprocess
import sys
import time
import traceback
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from ray.util.scheduling_strategies import (
import ray
from ray._private.gcs_utils import GcsAioClient
from ray._private.utils import run_background_task
import ray._private.ray_constants as ray_constants
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
from ray.actor import ActorHandle
from ray.dashboard.consts import (
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.utils import file_tail_iterator
from ray.exceptions import ActorUnschedulableError, RuntimeEnvSetupError
from ray.job_submission import JobStatus
from ray._private.event.event_logger import get_event_logger
from ray.core.generated.event_pb2 import Event
class JobLogStorageClient:
    """
    Disk storage for stdout / stderr of driver script logs.
    """
    NUM_LOG_LINES_ON_ERROR = 10
    MAX_LOG_SIZE = 20000

    def get_logs(self, job_id: str) -> str:
        try:
            with open(self.get_log_file_path(job_id), 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ''

    def tail_logs(self, job_id: str) -> Iterator[List[str]]:
        return file_tail_iterator(self.get_log_file_path(job_id))

    def get_last_n_log_lines(self, job_id: str, num_log_lines=NUM_LOG_LINES_ON_ERROR) -> str:
        """
        Returns the last MAX_LOG_SIZE (20000) characters in the last
        `num_log_lines` lines.

        Args:
            job_id: The id of the job whose logs we want to return
            num_log_lines: The number of lines to return.
        """
        log_tail_iter = self.tail_logs(job_id)
        log_tail_deque = deque(maxlen=num_log_lines)
        for lines in log_tail_iter:
            if lines is None:
                break
            else:
                for line in lines:
                    log_tail_deque.append(line)
        return ''.join(log_tail_deque)[-self.MAX_LOG_SIZE:]

    def get_log_file_path(self, job_id: str) -> Tuple[str, str]:
        """
        Get the file path to the logs of a given job. Example:
            /tmp/ray/session_date/logs/job-driver-{job_id}.log
        """
        return os.path.join(ray._private.worker._global_node.get_logs_dir_path(), JOB_LOGS_PATH_TEMPLATE.format(submission_id=job_id))