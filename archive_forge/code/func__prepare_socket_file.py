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
def _prepare_socket_file(self, socket_path: str, default_prefix: str):
    """Prepare the socket file for raylet and plasma.

        This method helps to prepare a socket file.
        1. Make the directory if the directory does not exist.
        2. If the socket file exists, do nothing (this just means we aren't the
           first worker on the node).

        Args:
            socket_path: the socket file to prepare.
        """
    result = socket_path
    is_mac = sys.platform.startswith('darwin')
    if sys.platform == 'win32':
        if socket_path is None:
            result = f'tcp://{self._localhost}:{self._get_unused_port()}'
    else:
        if socket_path is None:
            result = self._make_inc_temp(prefix=default_prefix, directory_name=self._sockets_dir)
        else:
            try_to_create_directory(os.path.dirname(socket_path))
        maxlen = (104 if is_mac else 108) - 1
        if len(result.split('://', 1)[-1].encode('utf-8')) > maxlen:
            raise OSError(f'AF_UNIX path length cannot exceed {maxlen} bytes: {result!r}')
    return result