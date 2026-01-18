from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
class PartitionLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float
    overall_latency_sec: float