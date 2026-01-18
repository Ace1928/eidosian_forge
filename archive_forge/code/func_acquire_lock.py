import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def acquire_lock(file_path):
    mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
    try:
        fd = os.open(file_path, mode)
        os.chmod(file_path, 511)
        max_lock_iter = 600
        for _ in range(max_lock_iter):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                return fd
            time.sleep(10)
        raise TimeoutError(f'Acquiring lock on file {file_path} timeout.')
    except Exception:
        os.close(fd)