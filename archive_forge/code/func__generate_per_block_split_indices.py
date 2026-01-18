import itertools
import logging
from typing import Iterable, List, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.types import ObjectRef
def _generate_per_block_split_indices(num_rows_per_block: List[int], split_indices: List[int]) -> List[List[int]]:
    """Given num rows per block and valid split indices, generate per block split indices.

    Args:
        num_rows_per_block: num of rows per block.
        split_indices: The (global) indices at which to split the blocks.
    Returns:
        Per block split indices indicates each input block's split point(s).
    """
    per_block_split_indices = []
    current_input_block_id = 0
    current_block_split_indices = []
    current_block_global_offset = 0
    current_index_id = 0
    while current_index_id < len(split_indices):
        split_index = split_indices[current_index_id]
        current_block_row = num_rows_per_block[current_input_block_id]
        if split_index - current_block_global_offset <= current_block_row:
            current_block_split_indices.append(split_index - current_block_global_offset)
            current_index_id += 1
            continue
        per_block_split_indices.append(current_block_split_indices)
        current_block_split_indices = []
        current_block_global_offset += num_rows_per_block[current_input_block_id]
        current_input_block_id += 1
    while len(per_block_split_indices) < len(num_rows_per_block):
        per_block_split_indices.append(current_block_split_indices)
        current_block_split_indices = []
    return per_block_split_indices