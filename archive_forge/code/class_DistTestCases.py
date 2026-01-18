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
@dataclass
class DistTestCases:
    skip_collective = {}
    skip_collective['allgather_coalesced'] = {'nccl', 'mpi', 'ucc'}
    skip_collective['reduce'] = set()
    skip_collective['sendrecv anysource'] = {'nccl', 'ucc'}
    skip_collective['cpu barrier'] = {'nccl', 'ucc'}
    backend_feature = {}
    backend_feature['gpu'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['cuda'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['ddp'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['subgroup'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['plugin'] = set()