import collections
import itertools
from contextlib import nullcontext
from typing import Any, Callable, Iterator, Optional, TypeVar
import ray
from ray.data._internal.block_batching.interfaces import BlockPrefetcher
from ray.data._internal.block_batching.util import (
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
def _prefetch_blocks(block_ref_iter: Iterator[ObjectRef[Block]], prefetcher: BlockPrefetcher, num_blocks_to_prefetch: int, eager_free: bool=False, stats: Optional[DatasetStats]=None) -> Iterator[ObjectRef[Block]]:
    """Given an iterable of Block Object References, returns an iterator
    over these object reference while prefetching `num_block_to_prefetch`
    blocks in advance.

    Args:
        block_ref_iter: An iterator over block object references.
        num_blocks_to_prefetch: The number of blocks to prefetch ahead of the
            current block during the scan.
        stats: Dataset stats object used to store block wait time.
    """
    if num_blocks_to_prefetch == 0:
        for block_ref in block_ref_iter:
            yield block_ref
            trace_deallocation(block_ref, 'block_batching._prefetch_blocks', free=eager_free)
    window_size = num_blocks_to_prefetch
    sliding_window = collections.deque(itertools.islice(block_ref_iter, window_size), maxlen=window_size)
    with stats.iter_wait_s.timer() if stats else nullcontext():
        prefetcher.prefetch_blocks(list(sliding_window))
    while sliding_window:
        block_ref = sliding_window.popleft()
        try:
            sliding_window.append(next(block_ref_iter))
            with stats.iter_wait_s.timer() if stats else nullcontext():
                prefetcher.prefetch_blocks(list(sliding_window))
        except StopIteration:
            pass
        yield block_ref
        trace_deallocation(block_ref, 'block_batching._prefetch_blocks', free=eager_free)
    prefetcher.stop()