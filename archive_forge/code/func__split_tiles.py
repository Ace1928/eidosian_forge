import cupy
import numpy
def _split_tiles(array, nr_splits):
    axis = nr_splits.argmax()
    ind = nr_splits[axis]
    if isinstance(array, cupy.ndarray):
        arrs = [cupy.ascontiguousarray(a) for a in cupy.array_split(array, int(ind), int(axis))]
    else:
        arrs = numpy.array_split(array, ind, axis)
    return (axis, arrs)