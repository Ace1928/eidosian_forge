from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
@dataclasses.dataclass
class MemoryPlanner:
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """
    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: Optional[List[BufferGroup]] = None

    def plan(self, lines: List[Any]) -> List[Any]:
        """Call all the memory planning passes in sequence"""
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate_groups()
        self.mark_first_last_usage(lines)
        return lines

    def drop_removed_buffers(self, lines):
        """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
        for i, line in enumerate(lines):
            if isinstance(line, (AllocateLine, FreeIfNotReusedLine, ReuseLine)):
                if line.node.get_name() in V.graph.removed_buffers:
                    lines[i] = NullLine(self.wrapper)

    def compute_buffer_groups(self, lines):
        """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
        name_to_group = {}
        for line in lines:
            if isinstance(line, AllocateLine):
                name = line.node.get_name()
                assert name not in name_to_group
                name_to_group[name] = BufferGroup(line.node)
            elif isinstance(line, ReuseLine):
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                assert new_name not in name_to_group
                if old_name in name_to_group:
                    name_to_group[old_name].names.append(new_name)
                    name_to_group[new_name] = name_to_group[old_name]
        outputs = set(V.graph.get_output_names())
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        for group in unique_groups:
            group.is_output = any((x in outputs for x in group.names))
        assert self.buffer_groups is None
        self.buffer_groups = unique_groups
        return name_to_group

    def convert_to_pool_lines(self, lines):
        """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
        name_to_group = self.compute_buffer_groups(lines)
        for i, line in enumerate(lines):
            if isinstance(line, AllocateLine):
                if line.node.get_name() in name_to_group:
                    lines[i] = AllocFromPoolLine(self.wrapper, name_to_group[line.node.get_name()])
            elif isinstance(line, FreeIfNotReusedLine):
                assert not line.is_reused
                if line.node.get_name() in name_to_group:
                    lines[i] = DeallocFromPoolLine(self.wrapper, name_to_group[line.node.get_name()])
            elif isinstance(line, ReuseLine):
                if line.node.get_name() in name_to_group:
                    line.delete_old = False

    def compute_live_ranges(self, lines):
        """Populate every BufferGroup.live_ranges field based on first/last usage"""
        timestep = 0
        worklist = collections.deque(lines)
        while worklist:
            if isinstance(worklist[0], MemoryPlanningLine):
                timestep += 1
                while worklist and isinstance(worklist[0], MemoryPlanningLine):
                    line = worklist.popleft()
                    if isinstance(line, PoolMemoryPlanningLine):
                        line.group.update_usage(timestep)
                        line.timestep = timestep
            else:
                worklist.popleft()
        timestep += 1
        assert self.buffer_groups is not None
        for group in self.buffer_groups:
            if group.is_output:
                group.update_usage(timestep)

    def allocate_groups(self):
        """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
        assert config.memory_pool in ('none', 'intermediates', 'outputs', 'combined')
        assert self.buffer_groups is not None
        for group in self.buffer_groups:
            group.make_allocation()
        outputs: List[Allocation] = []
        intermediates: List[Allocation] = []
        for group in self.buffer_groups:
            assert group.allocation
            if group.is_output and config.memory_pool != 'combined':
                outputs.append(group.allocation)
            else:
                intermediates.append(group.allocation)
        for block in sorted(outputs, key=lambda x: (x.size_hint, -len(x.live_range))):
            self.pools.allocate_output(block)
        for block in sorted(intermediates, key=lambda x: (-x.size_hint, -len(x.live_range))):
            self.pools.allocate(block)
        self.pools.finalize()

    def mark_first_last_usage(self, lines):
        """
        Populate the AllocFromPoolLine.is_first_pool_usage and
        DeallocFromPoolLine.is_last_pool_usage fields so that pools
        are created/destroyed.
        """
        seen = set()
        for line in lines:
            if isinstance(line, AllocFromPoolLine):
                assert line.group.allocation
                pool = line.group.allocation.pool
                assert pool is not None
                if pool not in seen:
                    line.is_first_pool_usage = True
                    seen.add(pool)
        seen = set()
        for line in reversed(lines):
            if isinstance(line, DeallocFromPoolLine):
                assert line.group.allocation
                pool = line.group.allocation.pool
                assert pool is not None
                if pool not in seen:
                    line.is_last_pool_usage = pool.root.get_live_ranges().end <= line.timestep
                    seen.add(pool)