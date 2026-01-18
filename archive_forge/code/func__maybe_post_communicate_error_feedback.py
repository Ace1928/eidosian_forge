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
def _maybe_post_communicate_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    ef_unroll_rec = self._create_event_recorder('Sync and error feedback unroll rec')
    if self._should_use_error_feedback(fp16_fp32_list):
        with torch.no_grad():
            for p, p_fp32 in fp16_fp32_list:
                p_fp32 += p.float()
    ef_unroll_rec.stop()
    self.logger.debug('Error feedback unroll completed')