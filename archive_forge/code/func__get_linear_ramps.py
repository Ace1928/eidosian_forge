import numbers
import numpy
import cupy
def _get_linear_ramps(padded, axis, width_pair, end_value_pair):
    """Constructs linear ramps for an empty-padded array along a given axis.

    Args:
      padded(cupy.ndarray): Empty-padded array.
      axis(int): Dimension in which the ramps are constructed.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
      end_value_pair((scalar, scalar)): End values for the linear ramps which
          form the edge of the fully padded array. These values are included in
          the linear ramps.
    """
    edge_pair = _get_edges(padded, axis, width_pair)
    left_ramp = cupy.linspace(start=end_value_pair[0], stop=edge_pair[0].squeeze(axis), num=width_pair[0], endpoint=False, dtype=padded.dtype, axis=axis)
    right_ramp = cupy.linspace(start=end_value_pair[1], stop=edge_pair[1].squeeze(axis), num=width_pair[1], endpoint=False, dtype=padded.dtype, axis=axis)
    right_ramp = right_ramp[_slice_at_axis(slice(None, None, -1), axis)]
    return (left_ramp, right_ramp)