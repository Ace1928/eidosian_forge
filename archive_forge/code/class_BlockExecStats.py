import collections
import os
import time
from dataclasses import dataclass
from typing import (
import numpy as np
import ray
from ray import DynamicObjectRefGenerator
from ray.data._internal.util import _check_pyarrow_version, _truncated_repr
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI
import psutil
@DeveloperAPI
class BlockExecStats:
    """Execution stats for this block.

    Attributes:
        wall_time_s: The wall-clock time it took to compute this block.
        cpu_time_s: The CPU time it took to compute this block.
        node_id: A unique id for the node that computed this block.
    """

    def __init__(self):
        self.start_time_s: Optional[float] = None
        self.end_time_s: Optional[float] = None
        self.wall_time_s: Optional[float] = None
        self.cpu_time_s: Optional[float] = None
        self.node_id = ray.runtime_context.get_runtime_context().get_node_id()
        self.max_rss_bytes: int = 0

    @staticmethod
    def builder() -> '_BlockExecStatsBuilder':
        return _BlockExecStatsBuilder()

    def __repr__(self):
        return repr({'wall_time_s': self.wall_time_s, 'cpu_time_s': self.cpu_time_s, 'node_id': self.node_id})