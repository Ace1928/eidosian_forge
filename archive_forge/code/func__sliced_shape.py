import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def _sliced_shape(self, shapes, i, major_axis):
    """Get the sliced shapes for the i-th executor.

        Parameters
        ----------
        shapes : list of (str, tuple)
            The original (name, shape) pairs.
        i : int
            Which executor we are dealing with.
        """
    sliced_shapes = []
    for desc, axis in zip(shapes, major_axis):
        shape = list(desc.shape)
        if axis >= 0:
            shape[axis] = self.slices[i].stop - self.slices[i].start
        sliced_shapes.append(DataDesc(desc.name, tuple(shape), desc.dtype, desc.layout))
    return sliced_shapes