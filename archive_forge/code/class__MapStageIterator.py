import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class _MapStageIterator:

    def __init__(self, input_blocks_list, shuffle_map, map_args):
        self._input_blocks_list = input_blocks_list
        self._shuffle_map = shuffle_map
        self._map_args = map_args
        self._mapper_idx = 0
        self._map_results = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self._input_blocks_list:
            raise StopIteration
        block = self._input_blocks_list.pop(0)
        map_result = self._shuffle_map.remote(self._mapper_idx, block, *self._map_args)
        metadata_ref = map_result.pop(-1)
        self._map_results.append(map_result)
        self._mapper_idx += 1
        return metadata_ref

    def pop_map_results(self) -> List[List[ObjectRef]]:
        map_results = self._map_results
        self._map_results = []
        return map_results