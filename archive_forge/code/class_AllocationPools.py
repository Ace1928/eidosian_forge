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
class AllocationPools:
    """
    Collection of many AllocationPool objects grouped by device.
    """
    device_to_pools: Dict[torch.device, List[AllocationPool]] = dataclasses.field(default_factory=dict)

    def get_pools(self, block):
        if block.device not in self.device_to_pools:
            self.device_to_pools[block.device] = []
        return self.device_to_pools[block.device]

    def allocate(self, block: Allocation):
        pools = self.get_pools(block)
        for pool in pools:
            if pool.allocate(block, is_last=pool is pools[-1]):
                return
        pools.append(AllocationPool(block.device, TemporalSplit([block]), can_expand=config.memory_pool != 'none'))
        block.mark_allocated()

    def allocate_output(self, block: Allocation):
        """Outputs get different pools so memory gets freed properly"""
        pools = self.get_pools(block)
        if pools and config.memory_pool in ('outputs', 'combined'):
            pools[-1].allocate_at_end(block)
        else:
            block.mark_allocated()
            pools.append(AllocationPool(block.device, TemporalSplit([block]), can_expand=config.memory_pool == 'combined'))

    def finalize(self):
        """Called at the end of allocation process"""
        for i, pool in enumerate(itertools.chain.from_iterable(self.device_to_pools.values())):
            pool.finalize(f'pool{i}')

    def pprint(self):
        for pool in itertools.chain.from_iterable(self.device_to_pools.values()):
            print()
            print(pool.name)
            print(pool.root.get_live_ranges())
            pprint.pprint(pool.root)