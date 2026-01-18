import typing
from typing import Any, Optional
def _index_for_subindex(a_idx: tuple[slice, ...], sub_idx: tuple[slice, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    assert len(a_idx) == len(sub_idx)
    return tuple((_index_for_subslice(a, sub, length) for a, sub, length in zip(a_idx, sub_idx, shape)))