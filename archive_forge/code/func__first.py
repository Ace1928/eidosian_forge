import cupy
def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position

    """
    return cupy.take_along_axis(arr, cupy.array(0, ndmin=arr.ndim), axis)