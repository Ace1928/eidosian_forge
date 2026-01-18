from enum import Enum
import functools
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.modules import Module
from .gossiper import Gossiper, PushPull, PushSum
from .graph_manager import GraphManager
from .graph_manager import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from .mixing_manager import MixingManager, UniformMixing
from .utils import (
from .utils.cuda_metering import EventRecorder, create_event_recorder
def _global_momentum_step(self, optimizer: torch.optim.Optimizer) -> None:
    """Performs the slow momentum step"""
    if not self.slowmo:
        return
    if not self.global_momentum_buffers_initialized:
        self._init_global_momentum_buffers(optimizer)
    if self.slowmo_memory_efficient:
        self._distributed_comm(optimizer, mode='gather')
    if self.is_current_node_a_slowmo_shard:
        self._perform_local_optimization(optimizer)
    if self.slowmo_memory_efficient:
        self._distributed_comm(optimizer, mode='scatter')