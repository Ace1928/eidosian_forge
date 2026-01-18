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
def _kill_process_type(self, process_type, allow_graceful: bool=False, check_alive: bool=True, wait: bool=False):
    """Kill a process of a given type.

        If the process type is PROCESS_TYPE_REDIS_SERVER, then we will kill all
        of the Redis servers.

        If the process was started in valgrind, then we will raise an exception
        if the process has a non-zero exit code.

        Args:
            process_type: The type of the process to kill.
            allow_graceful: Send a SIGTERM first and give the process
                time to exit gracefully. If that doesn't work, then use
                SIGKILL. We usually want to do this outside of tests.
            check_alive: If true, then we expect the process to be alive
                and will raise an exception if the process is already dead.
            wait: If true, then this method will not return until the
                process in question has exited.

        Raises:
            This process raises an exception in the following cases:
                1. The process had already died and check_alive is true.
                2. The process had been started in valgrind and had a non-zero
                   exit code.
        """
    with self.removal_lock:
        self._kill_process_impl(process_type, allow_graceful=allow_graceful, check_alive=check_alive, wait=wait)