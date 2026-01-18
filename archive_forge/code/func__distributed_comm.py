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
def _distributed_comm(self, optimizer: torch.optim.Optimizer, mode: str) -> None:
    """Performs the communication needed for the efficient SlowMo implementation"""
    offset = 0
    slowmo_comm_lists: List[List[torch.Tensor]] = [[] for _ in range(self.slowmo_num_shards)]
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                numel = p.numel()
                if mode == 'gather':
                    p /= self.process_world_size
                current_start = offset
                while current_start < offset + numel:
                    main_node = current_start // self.world_portion_length
                    main_node_end = (main_node + 1) * self.world_portion_length
                    current_end = min(offset + numel, main_node_end)
                    p_start = current_start - offset
                    p_end = current_end - offset
                    slowmo_comm_lists[main_node].append(p.view(-1)[p_start:p_end])
                    current_start = current_end
                offset += numel
        for slowmo_rank, slowmo_comm_list in enumerate(slowmo_comm_lists):
            if mode == 'gather':
                communication_op = functools.partial(dist.reduce, dst=slowmo_rank)
            elif mode == 'scatter':
                communication_op = functools.partial(dist.broadcast, src=slowmo_rank)
            communicate(slowmo_comm_list, communication_op)