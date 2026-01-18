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
class PyTorchProfiler_CUDAOnly(PyTorchProfiler):
    ACTIVITIES = [torch.profiler.ProfilerActivity.CUDA]