import collections
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
def _map_block_nosplit(block_fn: BlockTransform, input_files: List[str], fn: Optional[UserDefinedFunction], num_blocks: int, *blocks_and_fn_args: Union[Block, Any], **fn_kwargs) -> Tuple[Block, BlockMetadata]:
    stats = BlockExecStats.builder()
    builder = DelegatingBlockBuilder()
    blocks, fn_args = (blocks_and_fn_args[:num_blocks], blocks_and_fn_args[num_blocks:])
    if fn is not None:
        fn_args = (fn,) + fn_args
    for new_block in block_fn(blocks, *fn_args, **fn_kwargs):
        builder.add_block(new_block)
    new_block = builder.build()
    accessor = BlockAccessor.for_block(new_block)
    return (new_block, accessor.get_metadata(input_files=input_files, exec_stats=stats.build()))