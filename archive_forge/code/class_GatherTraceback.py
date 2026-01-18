import torch
from torch.utils._pytree import tree_map
from typing import Iterator, List, Optional
import logging
import contextlib
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks
class GatherTraceback(logging.Filter):

    def __init__(self, python=True, script=True, cpp=False):
        self.python = python
        self.script = script
        self.cpp = cpp

    def filter(self, record):
        record.traceback = gather_traceback(python=self.python, script=self.script, cpp=self.cpp)
        return True