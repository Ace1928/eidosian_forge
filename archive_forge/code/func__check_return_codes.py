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
def _check_return_codes(cls, failed_ranks, timeout, fn):
    error_msg = ''
    skip_code = -1
    for rank, exc_info in failed_ranks:
        exc = exc_info[1]
        if isinstance(exc, unittest.SkipTest):
            logger.info('Thread %s skipping test %s for following reason: %s', rank, fn, str(exc))
            if skip_code < 0:
                skip_code = TEST_SKIPS['generic'].exit_code
        elif isinstance(exc, TimeoutError):
            msg = f'Thread {rank} terminated or timed out after {timeout} seconds\n'
            logger.error(msg)
            raise RuntimeError(msg)
        elif isinstance(exc, Exception):
            msg = ''.join(traceback.format_exception(*exc_info))
            logger.error('Caught exception: \n%s exiting thread %s', msg, rank)
            error_msg += f'Thread {rank} exited with exception:\n{msg}\n'
        elif isinstance(exc, SystemExit):
            if type(exc.code) == int and skip_code < 0:
                skip_code = exc.code
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
    if skip_code > 0:
        for skip in TEST_SKIPS.values():
            if skip_code == skip.exit_code:
                if IS_SANDCASTLE:
                    logger.info('Skipping %s on sandcastle for the following reason: %s', fn, skip.message)
                    return
                else:
                    raise unittest.SkipTest(skip.message)