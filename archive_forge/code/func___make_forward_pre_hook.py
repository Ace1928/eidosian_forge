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
def __make_forward_pre_hook(self) -> Callable[..., None]:
    self.logger.debug('making forward pre-hook')

    def hook(*unused: Any) -> None:
        """Query gossip queue and de-bias during forward pass"""
        self._sync_buffers()
        if self.sgp:
            if self.gossip_enable and self.overlap and (not self._skip_averaging_memory_efficient_slowmo()):
                self._sgp_transfer_params()
            self._sgp_unbias()
    return hook