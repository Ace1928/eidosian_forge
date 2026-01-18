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
def _maybe_perform_sgp(self) -> None:
    sgp_rec = self._create_event_recorder('SGP')
    if self._should_perform_sgp():
        if not self._should_allreduce_params():
            self._sgp_transfer_params()
            self._sgp_query_gossip_queue()
            torch.cuda.synchronize()
        self.logger.debug('SGP completed')
    sgp_rec.stop()