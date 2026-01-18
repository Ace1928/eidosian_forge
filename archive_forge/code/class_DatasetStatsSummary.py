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
@DeveloperAPI
@dataclass
class DatasetStatsSummary:
    stages_stats: List['StageStatsSummary']
    iter_stats: 'IterStatsSummary'
    parents: List['DatasetStatsSummary']
    number: int
    dataset_uuid: str
    time_total_s: float
    base_name: str
    extra_metrics: Dict[str, Any]
    global_bytes_spilled: int
    global_bytes_restored: int
    dataset_bytes_spilled: int

    def to_string(self, already_printed: Optional[Set[str]]=None, include_parent: bool=True, add_global_stats=True) -> str:
        """Return a human-readable summary of this Dataset's stats.

        Args:
            already_printed: Set of stage IDs that have already had its stats printed
            out.
            include_parent: If true, also include parent stats summary; otherwise, only
            log stats of the latest stage.
            add_global_stats: If true, includes global stats to this summary.
        Returns:
            String with summary statistics for executing the Dataset.
        """
        if already_printed is None:
            already_printed = set()
        out = ''
        if self.parents and include_parent:
            for p in self.parents:
                parent_sum = p.to_string(already_printed, add_global_stats=False)
                if parent_sum:
                    out += parent_sum
                    out += '\n'
        stage_stats_summary = None
        if len(self.stages_stats) == 1:
            stage_stats_summary = self.stages_stats[0]
            stage_name = stage_stats_summary.stage_name
            stage_uuid = self.dataset_uuid + stage_name
            out += 'Stage {} {}: '.format(self.number, stage_name)
            if stage_uuid in already_printed:
                out += '[execution cached]\n'
            else:
                already_printed.add(stage_uuid)
                out += str(stage_stats_summary)
        elif len(self.stages_stats) > 1:
            rounded_total = round(self.time_total_s, 2)
            if rounded_total <= 0:
                rounded_total = 0
            out += 'Stage {} {}: executed in {}s\n'.format(self.number, self.base_name, rounded_total)
            for n, stage_stats_summary in enumerate(self.stages_stats):
                stage_name = stage_stats_summary.stage_name
                stage_uuid = self.dataset_uuid + stage_name
                out += '\n'
                out += '\tSubstage {} {}: '.format(n, stage_name)
                if stage_uuid in already_printed:
                    out += '\t[execution cached]\n'
                else:
                    already_printed.add(stage_uuid)
                    out += str(stage_stats_summary)
        if self.extra_metrics:
            indent = '\t' if stage_stats_summary and stage_stats_summary.is_substage else ''
            out += indent
            out += '* Extra metrics: ' + str(self.extra_metrics) + '\n'
        out += str(self.iter_stats)
        if len(self.stages_stats) > 0 and add_global_stats:
            mb_spilled = round(self.global_bytes_spilled / 1000000.0)
            mb_restored = round(self.global_bytes_restored / 1000000.0)
            if mb_spilled or mb_restored:
                out += '\nCluster memory:\n'
                out += '* Spilled to disk: {}MB\n'.format(mb_spilled)
                out += '* Restored from disk: {}MB\n'.format(mb_restored)
            dataset_mb_spilled = round(self.dataset_bytes_spilled / 1000000.0)
            if dataset_mb_spilled:
                out += '\nDataset memory:\n'
                out += '* Spilled to disk: {}MB\n'.format(dataset_mb_spilled)
        return out

    def __repr__(self, level=0) -> str:
        indent = leveled_indent(level)
        stage_stats = '\n'.join([ss.__repr__(level + 2) for ss in self.stages_stats])
        parent_stats = '\n'.join([ps.__repr__(level + 2) for ps in self.parents])
        extra_metrics = '\n'.join((f'{leveled_indent(level + 2)}{k}: {v},' for k, v in self.extra_metrics.items()))
        stage_stats = f'\n{stage_stats},\n{indent}   ' if stage_stats else ''
        parent_stats = f'\n{parent_stats},\n{indent}   ' if parent_stats else ''
        extra_metrics = f'\n{extra_metrics}\n{indent}   ' if extra_metrics else ''
        return f'{indent}DatasetStatsSummary(\n{indent}   dataset_uuid={self.dataset_uuid},\n{indent}   base_name={self.base_name},\n{indent}   number={self.number},\n{indent}   extra_metrics={{{extra_metrics}}},\n{indent}   stage_stats=[{stage_stats}],\n{indent}   iter_stats={self.iter_stats.__repr__(level + 1)},\n{indent}   global_bytes_spilled={self.global_bytes_spilled / 1000000.0}MB,\n{indent}   global_bytes_restored={self.global_bytes_restored / 1000000.0}MB,\n{indent}   dataset_bytes_spilled={self.dataset_bytes_spilled / 1000000.0}MB,\n{indent}   parents=[{parent_stats}],\n{indent})'

    def get_total_wall_time(self) -> float:
        parent_wall_times = [p.get_total_wall_time() for p in self.parents]
        parent_max_wall_time = max(parent_wall_times) if parent_wall_times else 0
        return parent_max_wall_time + sum((ss.wall_time.get('max', 0) for ss in self.stages_stats))

    def get_total_cpu_time(self) -> float:
        parent_sum = sum((p.get_total_cpu_time() for p in self.parents))
        return parent_sum + sum((ss.cpu_time.get('sum', 0) for ss in self.stages_stats))

    def get_max_heap_memory(self) -> float:
        parent_memory = [p.get_max_heap_memory() for p in self.parents]
        parent_max = max(parent_memory) if parent_memory else 0
        if not self.stages_stats:
            return parent_max
        return max(parent_max, *[ss.memory.get('max', 0) for ss in self.stages_stats])