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
@dataclass
class StageStatsSummary:
    stage_name: str
    is_substage: bool
    time_total_s: float
    block_execution_summary_str: str
    wall_time: Optional[Dict[str, float]] = None
    cpu_time: Optional[Dict[str, float]] = None
    memory: Optional[Dict[str, float]] = None
    output_num_rows: Optional[Dict[str, float]] = None
    output_size_bytes: Optional[Dict[str, float]] = None
    node_count: Optional[Dict[str, float]] = None

    @classmethod
    def from_block_metadata(cls, block_metas: List[BlockMetadata], stage_name: str, is_substage: bool) -> 'StageStatsSummary':
        """Calculate the stats for a stage from a given list of blocks,
        and generates a `StageStatsSummary` object with the results.

        Args:
            block_metas: List of `BlockMetadata` to calculate stats of
            stage_name: Name of stage associated with `blocks`
            is_substage: Whether this set of blocks belongs to a substage.
        Returns:
            A `StageStatsSummary` object initialized with the calculated statistics
        """
        exec_stats = [m.exec_stats for m in block_metas if m.exec_stats is not None]
        rounded_total = 0
        time_total_s = 0
        if is_substage:
            exec_summary_str = '{}/{} blocks executed\n'.format(len(exec_stats), len(block_metas))
        else:
            if exec_stats:
                earliest_start_time = min((s.start_time_s for s in exec_stats))
                latest_end_time = max((s.end_time_s for s in exec_stats))
                time_total_s = latest_end_time - earliest_start_time
                rounded_total = round(time_total_s, 2)
                if rounded_total <= 0:
                    rounded_total = 0
                exec_summary_str = '{}/{} blocks executed in {}s'.format(len(exec_stats), len(block_metas), rounded_total)
            else:
                exec_summary_str = ''
            if len(exec_stats) < len(block_metas):
                if exec_stats:
                    exec_summary_str += ', '
                num_inherited = len(block_metas) - len(exec_stats)
                exec_summary_str += '{}/{} blocks split from parent'.format(num_inherited, len(block_metas))
                if not exec_stats:
                    exec_summary_str += ' in {}s'.format(rounded_total)
            exec_summary_str += '\n'
        wall_time_stats = None
        if exec_stats:
            wall_time_stats = {'min': min([e.wall_time_s for e in exec_stats]), 'max': max([e.wall_time_s for e in exec_stats]), 'mean': np.mean([e.wall_time_s for e in exec_stats]), 'sum': sum([e.wall_time_s for e in exec_stats])}
        cpu_stats, memory_stats = (None, None)
        if exec_stats:
            cpu_stats = {'min': min([e.cpu_time_s for e in exec_stats]), 'max': max([e.cpu_time_s for e in exec_stats]), 'mean': np.mean([e.cpu_time_s for e in exec_stats]), 'sum': sum([e.cpu_time_s for e in exec_stats])}
            memory_stats_mb = [round(e.max_rss_bytes / (1024 * 1024), 2) for e in exec_stats]
            memory_stats = {'min': min(memory_stats_mb), 'max': max(memory_stats_mb), 'mean': int(np.mean(memory_stats_mb))}
        output_num_rows_stats = None
        output_num_rows = [m.num_rows for m in block_metas if m.num_rows is not None]
        if output_num_rows:
            output_num_rows_stats = {'min': min(output_num_rows), 'max': max(output_num_rows), 'mean': int(np.mean(output_num_rows)), 'sum': sum(output_num_rows)}
        output_size_bytes_stats = None
        output_size_bytes = [m.size_bytes for m in block_metas if m.size_bytes is not None]
        if output_size_bytes:
            output_size_bytes_stats = {'min': min(output_size_bytes), 'max': max(output_size_bytes), 'mean': int(np.mean(output_size_bytes)), 'sum': sum(output_size_bytes)}
        node_counts_stats = None
        if exec_stats:
            node_counts = collections.defaultdict(int)
            for s in exec_stats:
                node_counts[s.node_id] += 1
            node_counts_stats = {'min': min(node_counts.values()), 'max': max(node_counts.values()), 'mean': int(np.mean(list(node_counts.values()))), 'count': len(node_counts)}
        return StageStatsSummary(stage_name=stage_name, is_substage=is_substage, time_total_s=time_total_s, block_execution_summary_str=exec_summary_str, wall_time=wall_time_stats, cpu_time=cpu_stats, memory=memory_stats, output_num_rows=output_num_rows_stats, output_size_bytes=output_size_bytes_stats, node_count=node_counts_stats)

    def __str__(self) -> str:
        """For a given (pre-calculated) `StageStatsSummary` object (e.g. generated from
        `StageStatsSummary.from_block_metadata()`), returns a human-friendly string
        that summarizes stage execution statistics.

        Returns:
            String with summary statistics for executing the given stage.
        """
        indent = '\t' if self.is_substage else ''
        out = self.block_execution_summary_str
        wall_time_stats = self.wall_time
        if wall_time_stats:
            out += indent
            out += '* Remote wall time: {} min, {} max, {} mean, {} total\n'.format(fmt(wall_time_stats['min']), fmt(wall_time_stats['max']), fmt(wall_time_stats['mean']), fmt(wall_time_stats['sum']))
        cpu_stats = self.cpu_time
        if cpu_stats:
            out += indent
            out += '* Remote cpu time: {} min, {} max, {} mean, {} total\n'.format(fmt(cpu_stats['min']), fmt(cpu_stats['max']), fmt(cpu_stats['mean']), fmt(cpu_stats['sum']))
        memory_stats = self.memory
        if memory_stats:
            out += indent
            out += '* Peak heap memory usage (MiB): {} min, {} max, {} mean\n'.format(memory_stats['min'], memory_stats['max'], memory_stats['mean'])
        output_num_rows_stats = self.output_num_rows
        if output_num_rows_stats:
            out += indent
            out += '* Output num rows: {} min, {} max, {} mean, {} total\n'.format(output_num_rows_stats['min'], output_num_rows_stats['max'], output_num_rows_stats['mean'], output_num_rows_stats['sum'])
        output_size_bytes_stats = self.output_size_bytes
        if output_size_bytes_stats:
            out += indent
            out += '* Output size bytes: {} min, {} max, {} mean, {} total\n'.format(output_size_bytes_stats['min'], output_size_bytes_stats['max'], output_size_bytes_stats['mean'], output_size_bytes_stats['sum'])
        node_count_stats = self.node_count
        if node_count_stats:
            out += indent
            out += '* Tasks per node: {} min, {} max, {} mean; {} nodes used\n'.format(node_count_stats['min'], node_count_stats['max'], node_count_stats['mean'], node_count_stats['count'])
        return out

    def __repr__(self, level=0) -> str:
        """For a given (pre-calculated) `StageStatsSummary` object (e.g. generated from
        `StageStatsSummary.from_block_metadata()`), returns a human-friendly string
        that summarizes stage execution statistics.

        Returns:
            String with summary statistics for executing the given stage.
        """
        indent = leveled_indent(level)
        indent += leveled_indent(1) if self.is_substage else ''
        wall_time_stats = {k: fmt(v) for k, v in (self.wall_time or {}).items()}
        cpu_stats = {k: fmt(v) for k, v in (self.cpu_time or {}).items()}
        memory_stats = {k: fmt(v) for k, v in (self.memory or {}).items()}
        output_num_rows_stats = {k: fmt(v) for k, v in (self.output_num_rows or {}).items()}
        output_size_bytes_stats = {k: fmt(v) for k, v in (self.output_size_bytes or {}).items()}
        node_conut_stats = {k: fmt(v) for k, v in (self.node_count or {}).items()}
        out = f"{indent}StageStatsSummary(\n{indent}   stage_name='{self.stage_name}',\n{indent}   is_substage={self.is_substage},\n{indent}   time_total_s={fmt(self.time_total_s)},\n{indent}   block_execution_summary_str={self.block_execution_summary_str}{indent}   wall_time={wall_time_stats or None},\n{indent}   cpu_time={cpu_stats or None},\n{indent}   memory={memory_stats or None},\n{indent}   output_num_rows={output_num_rows_stats or None},\n{indent}   output_size_bytes={output_size_bytes_stats or None},\n{indent}   node_count={node_conut_stats or None},\n{indent})"
        return out