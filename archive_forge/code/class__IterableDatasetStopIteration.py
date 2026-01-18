import torch
import random
import os
import queue
from dataclasses import dataclass
from torch._utils import ExceptionWrapper
from typing import Optional, Union, TYPE_CHECKING
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS, HAS_NUMPY
@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    worker_id: int