from __future__ import annotations
import numpy as np
from dask.array.core import asarray, blockwise, einsum_lookup
from dask.utils import derived_from
Dask added an additional keyword-only argument ``split_every``.

    split_every: int >= 2 or dict(axis: int), optional
        Determines the depth of the recursive aggregation.
        Defaults to ``None`` which would let dask heuristically
        decide a good default.
    