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
class DynamoDistributedSingleProcTestCase(torch._dynamo.test_case.TestCase):
    """
    Test harness for single-process dynamo distributed tests,
    initializes dist process group.

    Prefer this for simple tests, as it's easier to debug.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(patch.dict(os.environ, {'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'}))
        cls.rank = 0
        cls.device = f'cuda:{cls.rank}'
        cls.device_ids = None if 'cuda' in cls.device else [cls.rank]
        c10d.init_process_group('nccl', rank=cls.rank, world_size=1)

    @classmethod
    def tearDownClass(cls):
        c10d.destroy_process_group()
        super().tearDownClass()