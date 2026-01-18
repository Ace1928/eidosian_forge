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
def _check_batch_size(blocks_and_meta: List[Tuple[ObjectRef[Block], BlockMetadata]], batch_size: int, name: str):
    """Log a warning if the provided batch size exceeds the configured target max block
    size.
    """
    batch_size_bytes = None
    for _, meta in blocks_and_meta:
        if meta.num_rows and meta.size_bytes:
            batch_size_bytes = math.ceil(batch_size * (meta.size_bytes / meta.num_rows))
            break
    context = DataContext.get_current()
    if batch_size_bytes is not None and batch_size_bytes > context.target_max_block_size:
        logger.warning(f'Requested batch size {batch_size} results in batches of {batch_size_bytes} bytes for {name} tasks, which is larger than the configured target max block size {context.target_max_block_size}. This may result in out-of-memory errors for certain workloads, and you may want to decrease your batch size or increase the configured target max block size, e.g.: from ray.data.context import DataContext; DataContext.get_current().target_max_block_size = 4_000_000_000')