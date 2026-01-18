import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, IO, AnyStr
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
from ray._private import storage
from ray._raylet import GcsClient, get_session_key_from_storage
from ray._private.resource_spec import ResourceSpec
from ray._private.services import serialize_config, get_address
from ray._private.utils import open_log, try_to_create_directory, try_to_symlink
def _kill_process_impl(self, process_type, allow_graceful=False, check_alive=True, wait=False):
    """See `_kill_process_type`."""
    if process_type not in self.all_processes:
        return
    process_infos = self.all_processes[process_type]
    if process_type != ray_constants.PROCESS_TYPE_REDIS_SERVER:
        assert len(process_infos) == 1
    for process_info in process_infos:
        process = process_info.process
        if process.poll() is not None:
            if check_alive:
                raise RuntimeError(f"Attempting to kill a process of type '{process_type}', but this process is already dead.")
            else:
                continue
        if process_info.use_valgrind:
            process.terminate()
            process.wait()
            if process.returncode != 0:
                message = f'Valgrind detected some errors in process of type {process_type}. Error code {process.returncode}.'
                if process_info.stdout_file is not None:
                    with open(process_info.stdout_file, 'r') as f:
                        message += '\nPROCESS STDOUT:\n' + f.read()
                if process_info.stderr_file is not None:
                    with open(process_info.stderr_file, 'r') as f:
                        message += '\nPROCESS STDERR:\n' + f.read()
                raise RuntimeError(message)
            continue
        if process_info.use_valgrind_profiler:
            os.kill(process.pid, signal.SIGINT)
            time.sleep(0.1)
        if allow_graceful:
            process.terminate()
            timeout_seconds = 1
            try:
                process.wait(timeout_seconds)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            process.kill()
            if wait:
                process.wait()
    del self.all_processes[process_type]