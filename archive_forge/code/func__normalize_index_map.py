import typing
from typing import Any, Optional
def _normalize_index_map(shape: tuple[int, ...], index_map: dict[int, Any]) -> dict[int, list[tuple[slice, ...]]]:
    new_index_map: dict[int, list[tuple[slice, ...]]] = {}
    for dev, idxs in index_map.items():
        if not isinstance(idxs, list):
            idxs = [idxs]
        idxs = [_normalize_index(shape, idx) for idx in idxs]
        idxs.sort()
        new_index_map[dev] = idxs
    return new_index_map