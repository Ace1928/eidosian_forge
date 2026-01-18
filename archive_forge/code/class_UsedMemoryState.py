import copy
import csv
import linecache
import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from datetime import datetime
from multiprocessing import Pipe, Process, Queue
from multiprocessing.connection import Connection
from typing import Callable, Iterable, List, NamedTuple, Optional, Union
from .. import AutoConfig, PretrainedConfig
from .. import __version__ as version
from ..utils import is_psutil_available, is_py3nvml_available, is_tf_available, is_torch_available, logging
from .benchmark_args_utils import BenchmarkArguments
class UsedMemoryState(NamedTuple):
    """
    `UsedMemoryState` are named tuples with the following fields:

        - 'frame': a `Frame` namedtuple (see below) storing information on the current tracing frame (current file,
          location in current file)
        - 'cpu_memory': CPU RSS memory state *before* executing the line
        - 'gpu_memory': GPU used memory *before* executing the line (sum for all GPUs or for only `gpus_to_trace` if
          provided)
    """
    frame: Frame
    cpu_memory: int
    gpu_memory: int