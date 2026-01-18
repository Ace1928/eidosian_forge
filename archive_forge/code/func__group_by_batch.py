import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _group_by_batch(shape: tuple[int, ...], index_map: dict[int, list[tuple[slice, ...]]]) -> dict[_BatchIdx, _BlockLocationMap]:
    location_maps: dict[_BatchIdx, _BlockLocationMap] = {}
    for dev, idxs in index_map.items():
        for chunk_i, idx in enumerate(idxs):
            idx_tuples = _convert_to_tuples(idx, shape)
            batch_idx, block_idx = (idx_tuples[:-2], idx_tuples[-2:])
            block_idx = typing.cast(_BlockIdx, block_idx)
            location_map = location_maps.setdefault(batch_idx, {})
            location = location_map.setdefault(block_idx, {})
            location[dev] = chunk_i
    return location_maps