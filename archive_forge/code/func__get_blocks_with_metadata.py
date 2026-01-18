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
def _get_blocks_with_metadata(self) -> Tuple[List[ObjectRef[Block]], List[BlockMetadata]]:
    """Get all underlying block futures and concrete metadata.

        This will block on the completion of the underlying read tasks and will fetch
        all block metadata outputted by those tasks.
        """
    block_refs, meta_refs = ([], [])
    for block_ref, meta_ref in self._iter_block_partition_refs():
        block_refs.append(block_ref)
        meta_refs.append(meta_ref)
    read_progress_bar = ProgressBar('Read progress', total=len(block_refs))
    unique_refs = list(set(block_refs))
    generators = read_progress_bar.fetch_until_complete(unique_refs)
    ref_to_blocks = {}
    ref_to_metadata = {}
    for ref, generator in zip(unique_refs, generators):
        refs_list = list(generator)
        meta = ray.get(refs_list.pop(-1))
        ref_to_blocks[ref] = refs_list
        ref_to_metadata[ref] = meta
    output_block_refs = []
    for idx, ref in enumerate(block_refs):
        output_block_refs += ref_to_blocks[ref]
        self._cached_metadata[idx] = ref_to_metadata[ref]
    return (output_block_refs, self._flatten_metadata(self._cached_metadata))