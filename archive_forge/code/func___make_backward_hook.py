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
def __make_backward_hook(self) -> Callable[..., None]:
    self.logger.debug('making backward hook')

    def hook(*unused: Any) -> None:
        if self.local_node_group is not None:
            grads = []
            for p in self.module.parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                p.grad.div_(self.nprocs_per_node)
                grads.append(p.grad)
            self.logger.debug('Gradients ready for syncing')
            communication_op = functools.partial(dist.all_reduce, group=self.local_node_group)
            communicate(grads, communication_op, self.logger)
            self.logger.debug('Gradient sync during backward pass in local_group complete')
        if self.sgp:
            self._sgp_ps_numerator()
            if self.gossip_enable and self.overlap and (not self._skip_averaging_memory_efficient_slowmo()):
                self._sgp_query_gossip_queue()

    def queue_hook(*unused: Any) -> None:
        Variable._execution_engine.queue_callback(hook)
    return queue_hook