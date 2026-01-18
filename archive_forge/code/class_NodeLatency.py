from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
class NodeLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float