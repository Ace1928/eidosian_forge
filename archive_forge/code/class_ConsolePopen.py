import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
class ConsolePopen(subprocess.Popen):
    if sys.platform == 'win32':

        def terminate(self):
            if isinstance(self.stdin, io.IOBase):
                self.stdin.close()
            if self._use_signals:
                self.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                super(ConsolePopen, self).terminate()

        def __init__(self, *args, **kwargs):
            new_pgroup = subprocess.CREATE_NEW_PROCESS_GROUP
            flags_to_add = 0
            if ray._private.utils.detect_fate_sharing_support():
                flags_to_add = new_pgroup
            flags_key = 'creationflags'
            if flags_to_add:
                kwargs[flags_key] = (kwargs.get(flags_key) or 0) | flags_to_add
            self._use_signals = kwargs[flags_key] & new_pgroup
            super(ConsolePopen, self).__init__(*args, **kwargs)