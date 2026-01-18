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
def _maybe_initialize_local_node_group(self, local_node_group: Optional[torch.distributed.ProcessGroup], process_rank: int, logical_world_size: int) -> None:
    if self.nprocs_per_node == 1:
        self.local_node_group = None
        return
    if local_node_group is not None:
        self.local_node_group = local_node_group
        return
    self.logger.debug('Initializing local process groups')
    for node in range(logical_world_size):
        node_processes_ranks = list(range(node * self.nprocs_per_node, (node + 1) * self.nprocs_per_node))
        new_local_group = create_process_group(node_processes_ranks)
        if process_rank in node_processes_ranks:
            self.local_node_group = new_local_group
    assert self.local_node_group is not None
    self.logger.debug('Initialization of local groups complete')