import math
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats, _get_or_create_stats_actor
from ray.data._internal.util import _split_list
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def _execute_read_task_split(i: int, task: ReadTask, context: DataContext, stats_uuid: str, stats_actor: ray.actor.ActorHandle) -> Iterable[Union[Block, List[BlockMetadata]]]:
    """Execute read task with dynamic block splitting.

    Returns an Iterable of blocks followed by their metadata.
    Example of return value for 3 blocks:
    (Block1, Block2, Block3, [BlockMetadata1, BlockMetadata2, BlockMetadata3])
    """
    DataContext._set_current(context)
    blocks = task()
    input_files = task.get_metadata().input_files
    blocks_metadata = []
    block_exec_stats = BlockExecStats.builder()
    for block in blocks:
        metadata = BlockAccessor.for_block(block).get_metadata(input_files=input_files, exec_stats=block_exec_stats.build())
        yield block
        blocks_metadata.append(metadata)
        block_exec_stats = BlockExecStats.builder()
    stats_actor.record_task.remote(stats_uuid, i, blocks_metadata)
    yield blocks_metadata