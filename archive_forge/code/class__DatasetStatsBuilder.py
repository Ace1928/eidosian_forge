import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class _DatasetStatsBuilder:
    """Helper class for building dataset stats.

    When this class is created, we record the start time. When build() is
    called with the final blocks of the new dataset, the time delta is
    saved as part of the stats."""

    def __init__(self, stage_name: str, parent: 'DatasetStats', override_start_time: Optional[float]):
        self.stage_name = stage_name
        self.parent = parent
        self.start_time = override_start_time or time.perf_counter()

    def build_multistage(self, stages: StatsDict) -> 'DatasetStats':
        stage_infos = {}
        for i, (k, v) in enumerate(stages.items()):
            capped_k = capfirst(k)
            if len(stages) > 1:
                if i == 0:
                    stage_infos[self.stage_name + capped_k] = v
                else:
                    stage_infos[self.stage_name.split('->')[-1] + capped_k] = v
            else:
                stage_infos[self.stage_name] = v
        stats = DatasetStats(stages=stage_infos, parent=self.parent, base_name=self.stage_name)
        stats.time_total_s = time.perf_counter() - self.start_time
        return stats

    def build(self, final_blocks: BlockList) -> 'DatasetStats':
        stats = DatasetStats(stages={self.stage_name: final_blocks.get_metadata()}, parent=self.parent)
        stats.time_total_s = time.perf_counter() - self.start_time
        return stats