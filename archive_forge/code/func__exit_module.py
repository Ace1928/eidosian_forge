import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
def _exit_module(self, name) -> None:
    torch.cuda.nvtx.range_pop()
    assert self.parents[-1] == name
    self.parents.pop()