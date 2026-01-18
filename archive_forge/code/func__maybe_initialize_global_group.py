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
def _maybe_initialize_global_group(self, global_group: Optional[torch.distributed.ProcessGroup], process_world_size: int) -> None:
    if global_group is None:
        all_processes = list(range(process_world_size))
        self.global_group = create_process_group(all_processes)
        self.logger.debug('Initialization of global group complete')
    else:
        self.global_group = global_group
    self.logger.debug('Global group set')
    self.process_group = self.global_group