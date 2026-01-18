import typing
from typing import Any, Optional
def _normalize_index(shape: tuple[int, ...], idx: Any) -> tuple[slice, ...]:
    if not isinstance(idx, tuple):
        idx = (idx,)
    ndim = len(shape)
    if len(idx) > ndim:
        raise IndexError(f'too many indices for array: array is {ndim}-dimensional, but {len(idx)} were indexed')
    idx = idx + (slice(None),) * (ndim - len(idx))
    new_idx = []
    for i in range(ndim):
        if isinstance(idx[i], int):
            if idx[i] >= shape[i]:
                raise IndexError(f'Index {idx[i]} is out of bounds for axis {i} with size {shape[i]}')
            new_idx.append(slice(idx[i], idx[i] + 1, 1))
        elif isinstance(idx[i], slice):
            start, stop, step = idx[i].indices(shape[i])
            if step <= 0:
                raise ValueError('Slice step must be positive.')
            if start == stop:
                raise ValueError(f'The index is empty on axis {i}')
            new_idx.append(slice(start, stop, step))
        else:
            raise ValueError(f'Invalid index on axis {i}')
    return tuple(new_idx)