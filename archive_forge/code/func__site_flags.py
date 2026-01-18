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
def _site_flags() -> List[str]:
    """Detect whether flags related to site packages are enabled for the current
    interpreter. To run Ray in hermetic build environments, it helps to pass these flags
    down to Python workers.
    """
    flags = []
    if _no_site():
        flags.append('-S')
    if _no_user_site():
        flags.append('-s')
    return flags