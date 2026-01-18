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
class NsightProfiler:
    """Profiler that triggers start of NSight profiler.

    NOTE: you need to ensure that the script running this code actually is running with
    ``nsys profile`` and also has a flag ``--capture-range=cudaProfilerApi`` so the
    capturing is performed by this profiler during certain steps.
    """

    def __init__(self, main_profiler: '_Profiler') -> None:
        self.main_profiler = main_profiler

    def __enter__(self):
        self.main_profiler._install_hooks()
        torch.cuda.profiler.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.profiler.stop()
        self.main_profiler._remove_hooks()

    def step(self) -> None:
        pass