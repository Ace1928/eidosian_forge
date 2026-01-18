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
@staticmethod
def _sgp_gossip_into_receive_buffer(send_buffer: List[torch.Tensor], gossiper: Gossiper, receive_buffer: List[torch.Tensor], gossip_ps_weight: torch.Tensor, gossip_lock: threading.Lock, dist_config: Dict[Any, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    out_msg = flatten_tensors(send_buffer)
    with gossip_lock:
        in_msg, ps_weight = gossiper.mix(out_msg, gossip_ps_weight)
        ps_factor = gossiper.mixing_weights['lo']
    with torch.no_grad():
        for r, g in zip(unflatten_tensors(in_msg, send_buffer), receive_buffer):
            if dist_config['cpu_comm']:
                g.copy_(r, non_blocking=True)
            else:
                g.copy_(r)
    return (ps_weight, ps_factor)