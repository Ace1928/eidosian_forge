import numpy as np
from warnings import warn
from ._sputils import isintlike
def _check_ellipsis(index):
    """Process indices with Ellipsis. Returns modified index."""
    if index is Ellipsis:
        return (slice(None), slice(None))
    if not isinstance(index, tuple):
        return index
    ellipsis_indices = [i for i, v in enumerate(index) if v is Ellipsis]
    if not ellipsis_indices:
        return index
    if len(ellipsis_indices) > 1:
        warn('multi-Ellipsis indexing is deprecated will be removed in v1.13.', DeprecationWarning, stacklevel=2)
    first_ellipsis = ellipsis_indices[0]
    if len(index) == 1:
        return (slice(None), slice(None))
    if len(index) == 2:
        if first_ellipsis == 0:
            if index[1] is Ellipsis:
                return (slice(None), slice(None))
            return (slice(None), index[1])
        return (index[0], slice(None))
    tail = []
    for v in index[first_ellipsis + 1:]:
        if v is not Ellipsis:
            tail.append(v)
    nd = first_ellipsis + len(tail)
    nslice = max(0, 2 - nd)
    return index[:first_ellipsis] + (slice(None),) * nslice + tuple(tail)