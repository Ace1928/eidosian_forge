from __future__ import annotations
from dask.array import core
def _tiledb_to_chunks(tiledb_array):
    schema = tiledb_array.schema
    return list((schema.domain.dim(i).tile for i in range(schema.ndim)))