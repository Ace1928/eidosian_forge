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
def _maybe_allreduce(self) -> None:
    localsgd_rec = self._create_event_recorder('Localsgd communication time')
    if self._should_allreduce_params():
        communication_op = functools.partial(dist.all_reduce, group=self.master_group)
        params = cast(List[torch.Tensor], list(self.parameters()))
        with torch.no_grad():
            for p in params:
                p.div_(self.logical_world_size)
        self.logger.debug('Params normalized before localsgd step')
        communicate(params, communication_op, self.logger)
        torch.cuda.synchronize()
        self.logger.debug('Allreduce completed')
    localsgd_rec.stop()