import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _broadcast_shapes(shapes, axis=None):
    """
    Broadcast shapes, ignoring incompatibility of specified axes
    """
    if not shapes:
        return shapes
    if axis is not None:
        axis = np.atleast_1d(axis)
        axis_int = axis.astype(int)
        if not np.array_equal(axis_int, axis):
            raise AxisError('`axis` must be an integer, a tuple of integers, or `None`.')
        axis = axis_int
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for row, shape in zip(new_shapes, shapes):
        row[len(row) - len(shape):] = shape
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = f'`axis` is out of bounds for array of dimension {n_dims}'
            raise AxisError(message)
        if len(np.unique(axis)) != len(axis):
            raise AxisError('`axis` must contain only distinct elements')
        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)
    new_shape = np.max(new_shapes, axis=0)
    new_shape *= new_shapes.all(axis=0)
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError('Array shapes are incompatible for broadcasting.')
    if axis is not None:
        new_axis = axis - np.arange(len(axis))
        new_shapes = [tuple(np.insert(new_shape, new_axis, removed_shape)) for removed_shape in removed_shapes]
        return new_shapes
    else:
        return tuple(new_shape)