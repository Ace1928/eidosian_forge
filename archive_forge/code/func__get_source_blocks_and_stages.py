import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def _get_source_blocks_and_stages(self) -> Tuple[BlockList, DatasetStats, List[Stage]]:
    """Get the source blocks, corresponding stats, and the stages for plan
        execution.

        If a computed snapshot exists and has not been cleared, return the snapshot
        blocks and stats; otherwise, return the input blocks and stats that the plan was
        created with.
        """
    stages = self._stages_after_snapshot.copy()
    if self._snapshot_blocks is not None:
        if not self._snapshot_blocks.is_cleared():
            blocks = self._snapshot_blocks
            stats = self._snapshot_stats
            self._clear_snapshot()
        else:
            blocks = self._in_blocks
            stats = self._in_stats
            stages = self._stages_before_snapshot + self._stages_after_snapshot
    else:
        blocks = self._in_blocks
        stats = self._in_stats
    return (blocks, stats, stages)