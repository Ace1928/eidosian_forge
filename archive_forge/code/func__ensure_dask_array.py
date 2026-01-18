import numpy
def _ensure_dask_array(array, chunks=None):
    import dask.array as da
    if isinstance(array, da.Array):
        return array
    return da.from_array(array, chunks=chunks)