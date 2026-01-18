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
def _exec_entrypoint(self, logs_path: str) -> subprocess.Popen:
    """
        Runs the entrypoint command as a child process, streaming stderr &
        stdout to given log files.

        Unix systems:
        Meanwhile we start a demon process and group driver
        subprocess in same pgid, such that if job actor dies, entire process
        group also fate share with it.

        Windows systems:
        A jobObject is created to enable fate sharing for the entire process group.

        Args:
            logs_path: File path on head node's local disk to store driver
                command's stdout & stderr.
        Returns:
            child_process: Child process that runs the driver command. Can be
                terminated or killed upon user calling stop().
        """
    with open(logs_path, 'w') as logs_file:
        child_process = subprocess.Popen(self._entrypoint, shell=True, start_new_session=True, stdout=logs_file, stderr=subprocess.STDOUT, preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT}) if sys.platform != 'win32' and os.environ.get('RAY_JOB_STOP_SIGNAL') == 'SIGINT' else None)
        parent_pid = os.getpid()
        child_pid = child_process.pid
        if sys.platform != 'win32':
            try:
                child_pgid = os.getpgid(child_pid)
            except ProcessLookupError:
                return child_process
            subprocess.Popen(f'while kill -s 0 {parent_pid}; do sleep 1; done; kill -9 -{child_pgid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'win32' and win32api:
            self._win32_job_object = win32job.CreateJobObject(None, '')
            win32_job_info = win32job.QueryInformationJobObject(self._win32_job_object, win32job.JobObjectExtendedLimitInformation)
            win32_job_info['BasicLimitInformation']['LimitFlags'] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(self._win32_job_object, win32job.JobObjectExtendedLimitInformation, win32_job_info)
            child_handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE | win32con.PROCESS_SET_QUOTA, False, child_pid)
            win32job.AssignProcessToJobObject(self._win32_job_object, child_handle)
        return child_process