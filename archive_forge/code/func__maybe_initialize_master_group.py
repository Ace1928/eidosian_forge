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
def _maybe_initialize_master_group(self, master_group: Optional[torch.distributed.ProcessGroup], process_rank: int, process_world_size: int, nprocs_per_node: int) -> None:
    if master_group is not None:
        self.master_group: Optional[torch.distributed.ProcessGroup] = master_group
        return
    if self.nprocs_per_node > 1:
        self.logger.debug('Initializing master process group')
        master_nodes = [i for i in range(process_world_size) if i % nprocs_per_node == 0]
        self.master_group = create_process_group(master_nodes) if len(master_nodes) > 1 else None
        if self.master_group is not None and process_rank in master_nodes:
            self.logger.debug('Initialization of master group complete')
    else:
        self.master_group = self.global_group