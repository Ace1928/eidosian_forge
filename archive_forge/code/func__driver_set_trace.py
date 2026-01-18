import errno
import inspect
import json
import logging
import os
import re
import select
import socket
import sys
import time
import traceback
import uuid
from pdb import Pdb
from typing import Callable
import setproctitle
import ray
from ray._private import ray_constants
from ray.experimental.internal_kv import _internal_kv_del, _internal_kv_put
from ray.util.annotations import DeveloperAPI
def _driver_set_trace():
    """The breakpoint hook to use for the driver.

    This disables Ray driver logs temporarily so that the PDB console is not
    spammed: https://github.com/ray-project/ray/issues/18172
    """
    if ray.util.ray_debugpy._is_ray_debugger_enabled():
        return ray.util.ray_debugpy.set_trace()
    print('*** Temporarily disabling Ray worker logs ***')
    ray._private.worker._worker_logs_enabled = False

    def enable_logging():
        print('*** Re-enabling Ray worker logs ***')
        ray._private.worker._worker_logs_enabled = True
    pdb = _PdbWrap(enable_logging)
    frame = sys._getframe().f_back
    pdb.set_trace(frame)