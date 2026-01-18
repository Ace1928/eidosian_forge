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
def _should_allreduce_params(self) -> bool:
    return self.sgp and self._should_perform_slowmo() and self.slowmo_sgp_average_params or (self._should_perform_localsgd() and (not self._skip_averaging_memory_efficient_slowmo()))