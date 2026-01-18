import collections
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import ray
from ray.data._internal.block_batching.interfaces import Batch, BlockPrefetcher
from ray.data._internal.block_batching.util import (
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import make_async_gen
from ray.data.block import Block, BlockMetadata, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
def _format_in_threadpool(batch_iter: Iterator[Batch], stats: DatasetStats, batch_format: Optional[str], collate_fn: Optional[Callable[[DataBatch], Any]], num_threadpool_workers: int) -> Iterator[Batch]:
    """Executes the batching, formatting, and collation logic in a threadpool.

    Args:
        logical_batch_iterator: An iterator over logical batches.
        stats: DatasetStats object to record timing and other statistics.
        batch_format: The format in which to return each batch.
            Specify "default" to use the current block format (promoting
            Arrow to pandas automatically), "pandas" to
            select ``pandas.DataFrame`` or "pyarrow" to select
            ``pyarrow.Table``, or None to use entire blocks
            as batches.
        collate_fn: A function to apply to each data batch before returning it.
        num_threadpool_workers: The number of threads to use in the threadpool.
    """

    def threadpool_computations_format_collate(batch_iter: Iterator[Batch]) -> Iterator[Batch]:
        formatted_batch_iter = format_batches(batch_iter, batch_format=batch_format, stats=stats)
        if collate_fn is not None:
            formatted_batch_iter = collate(formatted_batch_iter, collate_fn=collate_fn, stats=stats)
        yield from formatted_batch_iter
    if num_threadpool_workers > 0:
        collated_iter = make_async_gen(base_iterator=batch_iter, fn=threadpool_computations_format_collate, num_workers=num_threadpool_workers)
    else:
        collated_iter = threadpool_computations_format_collate(batch_iter)
    return collated_iter