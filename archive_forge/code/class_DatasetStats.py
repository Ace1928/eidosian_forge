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
class DatasetStats:
    """Holds the execution times for a given Dataset.

    This object contains a reference to the parent Dataset's stats as well,
    but not the Dataset object itself, to allow its blocks to be dropped from
    memory."""

    def __init__(self, *, stages: StatsDict, parent: Union[Optional['DatasetStats'], List['DatasetStats']], needs_stats_actor: bool=False, stats_uuid: str=None, base_name: str=None):
        """Create dataset stats.

        Args:
            stages: Dict of stages used to create this Dataset from the
                previous one. Typically one entry, e.g., {"map": [...]}.
            parent: Reference to parent Dataset's stats, or a list of parents
                if there are multiple.
            needs_stats_actor: Whether this Dataset's stats needs a stats actor for
                stats collection. This is currently only used for Datasets using a
                lazy datasource (i.e. a LazyBlockList).
            stats_uuid: The uuid for the stats, used to fetch the right stats
                from the stats actor.
            base_name: The name of the base operation for a multi-stage operation.
        """
        self.stages: StatsDict = stages
        if parent is not None and (not isinstance(parent, list)):
            parent = [parent]
        self.parents: List['DatasetStats'] = parent or []
        self.number: int = 0 if not self.parents else max((p.number for p in self.parents)) + 1
        self.base_name = base_name
        self.dataset_uuid: str = 'unknown_uuid'
        self.time_total_s: float = 0
        self.needs_stats_actor = needs_stats_actor
        self.stats_uuid = stats_uuid
        self.iter_wait_s: Timer = Timer()
        self.iter_get_s: Timer = Timer()
        self.iter_next_batch_s: Timer = Timer()
        self.iter_format_batch_s: Timer = Timer()
        self.iter_collate_batch_s: Timer = Timer()
        self.iter_finalize_batch_s: Timer = Timer()
        self.iter_total_blocked_s: Timer = Timer()
        self.iter_user_s: Timer = Timer()
        self.iter_total_s: Timer = Timer()
        self.extra_metrics = {}
        self.iter_blocks_local: int = 0
        self.iter_blocks_remote: int = 0
        self.iter_unknown_location: int = 0
        self.global_bytes_spilled: int = 0
        self.global_bytes_restored: int = 0
        self.dataset_bytes_spilled: int = 0

    @property
    def stats_actor(self):
        return _get_or_create_stats_actor()

    def child_builder(self, name: str, override_start_time: Optional[float]=None) -> _DatasetStatsBuilder:
        """Start recording stats for an op of the given name (e.g., map)."""
        return _DatasetStatsBuilder(name, self, override_start_time)

    def child_TODO(self, name: str) -> 'DatasetStats':
        """Placeholder for child ops not yet instrumented."""
        return DatasetStats(stages={name + '_TODO': []}, parent=self)

    @staticmethod
    def TODO():
        """Placeholder for ops not yet instrumented."""
        return DatasetStats(stages={'TODO': []}, parent=None)

    def to_summary(self) -> 'DatasetStatsSummary':
        """Generate a `DatasetStatsSummary` object from the given `DatasetStats`
        object, which can be used to generate a summary string."""
        if self.needs_stats_actor:
            ac = self.stats_actor
            stats_map, self.time_total_s = ray.get(ac.get.remote(self.stats_uuid))
            if len(stats_map.items()) == len(self.stages['Read']):
                self.stages['Read'] = []
                for _, blocks_metadata in sorted(stats_map.items()):
                    self.stages['Read'] += blocks_metadata
        stages_stats = []
        is_substage = len(self.stages) > 1
        for stage_name, metadata in self.stages.items():
            stages_stats.append(StageStatsSummary.from_block_metadata(metadata, stage_name, is_substage=is_substage))
        iter_stats = IterStatsSummary(self.iter_wait_s, self.iter_get_s, self.iter_next_batch_s, self.iter_format_batch_s, self.iter_collate_batch_s, self.iter_finalize_batch_s, self.iter_total_blocked_s, self.iter_user_s, self.iter_total_s, self.iter_blocks_local, self.iter_blocks_remote, self.iter_unknown_location)
        stats_summary_parents = []
        if self.parents is not None:
            stats_summary_parents = [p.to_summary() for p in self.parents]
        return DatasetStatsSummary(stages_stats, iter_stats, stats_summary_parents, self.number, self.dataset_uuid, self.time_total_s, self.base_name, self.extra_metrics, self.global_bytes_spilled, self.global_bytes_restored, self.dataset_bytes_spilled)