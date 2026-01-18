import faulthandler
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union
from unittest.mock import patch
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.multi_threaded_pg import (
@classmethod
def _join_threads(cls, threads, fn):
    timeout = TIMEOUT_DEFAULT
    try:
        for idx, thread in enumerate(threads):
            thread.join(max(0, timeout))
            if thread.is_alive():
                MultiThreadedTestCase.exception_queue.put((idx, (TimeoutError, TimeoutError(f'Rank failed to join in under {timeout} seconds'), None)))
        ProcessLocalGroup.reset()
        failed_ranks = []
        while not cls.exception_queue.empty():
            failure = cls.exception_queue.get()
            failed_ranks.append(failure)
    finally:
        _uninstall_threaded_pg()
    cls._check_return_codes(failed_ranks, timeout, fn)