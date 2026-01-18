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
def _sgp_ps_numerator(self) -> None:
    """Convert model params to ps-numerator"""
    if not self.is_sgp_ps_numerator:
        if not self.lazy_mixing:
            ps_weight = self.ps_weight
            with torch.no_grad():
                for p in self.module.parameters():
                    p.mul_(cast(torch.Tensor, ps_weight.type(p.dtype)))
        self.is_sgp_ps_numerator = True