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
class SaveForwardInputsModel(nn.Module):

    def __init__(self, forward_inputs: Dict[nn.Module, torch.Tensor], cast_forward_inputs: bool) -> None:
        super().__init__()
        self.c1 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.c2 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.forward_inputs = forward_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_inputs[self] = x
        return self.c2(self.c1(x))