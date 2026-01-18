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
def _maybe_sgp_overlap_pre_communicate_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    if self._should_perform_sgp_overlap() and fp16_fp32_list:
        if self.ef1 is None:
            self.ef1 = [p_fp32.clone().detach_() for _, p_fp32 in fp16_fp32_list]
        with torch.no_grad():
            assert self.ef1 is not None
            for ef1, (p_fp16, p_fp32) in zip(self.ef1, fp16_fp32_list):
                ef1.copy_(p_fp32 - p_fp16.float())