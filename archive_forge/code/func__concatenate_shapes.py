import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _concatenate_shapes(shapes, axis):
    """Given array shapes, return the resulting shape and slices prefixes.

    These help in nested concatenation.

    Returns
    -------
    shape: tuple of int
        This tuple satisfies::

            shape, _ = _concatenate_shapes([arr.shape for shape in arrs], axis)
            shape == concatenate(arrs, axis).shape

    slice_prefixes: tuple of (slice(start, end), )
        For a list of arrays being concatenated, this returns the slice
        in the larger array at axis that needs to be sliced into.

        For example, the following holds::

            ret = concatenate([a, b, c], axis)
            _, (sl_a, sl_b, sl_c) = concatenate_slices([a, b, c], axis)

            ret[(slice(None),) * axis + sl_a] == a
            ret[(slice(None),) * axis + sl_b] == b
            ret[(slice(None),) * axis + sl_c] == c

        These are called slice prefixes since they are used in the recursive
        blocking algorithm to compute the left-most slices during the
        recursion. Therefore, they must be prepended to rest of the slice
        that was computed deeper in the recursion.

        These are returned as tuples to ensure that they can quickly be added
        to existing slice tuple without creating a new tuple every time.

    """
    shape_at_axis = [shape[axis] for shape in shapes]
    first_shape = shapes[0]
    first_shape_pre = first_shape[:axis]
    first_shape_post = first_shape[axis + 1:]
    if any((shape[:axis] != first_shape_pre or shape[axis + 1:] != first_shape_post for shape in shapes)):
        raise ValueError('Mismatched array shapes in block along axis {}.'.format(axis))
    shape = first_shape_pre + (sum(shape_at_axis),) + first_shape[axis + 1:]
    offsets_at_axis = _accumulate(shape_at_axis)
    slice_prefixes = [(slice(start, end),) for start, end in zip([0] + offsets_at_axis, offsets_at_axis)]
    return (shape, slice_prefixes)