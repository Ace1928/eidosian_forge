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
def _wait_and_get_for_node_address(self, timeout_s: int=60) -> str:
    """Wait until the RAY_NODE_IP_FILENAME file is avialable.

        RAY_NODE_IP_FILENAME is created when a ray instance is started.

        Args:
            timeout_s: If the ip address is not found within this
                timeout, it will raise ValueError.
        Returns:
            The node_ip_address of the current session if it finds it
            within timeout_s.
        """
    for i in range(timeout_s):
        node_ip_address = ray._private.services.get_cached_node_ip_address(self.get_session_dir_path())
        if node_ip_address is not None:
            return node_ip_address
        time.sleep(1)
        if i % 10 == 0:
            logger.info(f"Can't find a `{ray_constants.RAY_NODE_IP_FILENAME}` file from {self.get_session_dir_path()}. Have you started Ray instsance using `ray start` or `ray.init`?")
    raise ValueError(f"Can't find a `{ray_constants.RAY_NODE_IP_FILENAME}` file from {self.get_session_dir_path()}. for {timeout_s} seconds. A ray instance hasn't started. Did you do `ray start` or `ray.init` on this host?")