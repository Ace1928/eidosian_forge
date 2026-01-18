from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
class dist_backend(str, Enum):
    UNDEFINED = 'undefined'
    TCP = 'tcp'
    MPI = 'mpi'
    GLOO = 'gloo'
    NCCL = 'nccl'