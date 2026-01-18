import functools
import time
from typing import List, Optional, Dict
import numpy as np
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.util.tf_export import tf_export
def _enumerate_core_locations(bounds: List[int], ring_bounds: List[int], axes: List[str], can_split_host_across_rings: bool, ring_size: int) -> List[_CoreLocation]:
    """Enumerates all possible core locations under the axis iteration order.

  Args:
    bounds: A list of 4 positive integers, upper bound values for x, y, z, core.
    ring_bounds: A list of 4 positive integers, upper bound values for ring size
      in x, y, z, core axes.
    axes: A permutation of ["x", "y", "z", "core"], the axis iteration order.
    can_split_host_across_rings: If true, devices attached to the same host may
      get assigned to different rings.
    ring_size: Number of devices in a ring, only for argument validation.

  Returns:
    A list of all CoreLocation objects defined in a TPU slice of shape `bounds`,
    sorted by axis iteration order specified by `axes`.

    For example, given bounds=[2, 2, 1, 2] and axes=["core", "z", "y", "x"],
    return 8 core locations expressed in (x, y, z, core) format but iterated in
    core -> z -> y -> x order (fatest to slowest varying):

    [_CoreLocation(0, 0, 0, 0),
     _CoreLocation(0, 0, 0, 1),
     _CoreLocation(0, 1, 0, 0),
     _CoreLocation(0, 1, 0, 1),
     _CoreLocation(1, 0, 0, 0),
     _CoreLocation(1, 0, 0, 1),
     _CoreLocation(1, 1, 0, 0),
     _CoreLocation(1, 1, 0, 1)]

  Raises:
    ValueError: If ring_size cannot be fulfilled without splitting hosts.
  """
    num_cores_per_chip = bounds[3]
    if num_cores_per_chip != 1 and num_cores_per_chip != 2:
        raise ValueError('Unsupported TPU slice size: %s' % bounds)
    axes = [{'x': 0, 'y': 1, 'z': 2, 'core': 3}[axis] for axis in axes]
    bounds = [bounds[i] for i in axes]
    if can_split_host_across_rings:
        host_bounds = [1, 1, 1, 1]
    elif np.prod(bounds) <= 2:
        host_bounds = [[1, 1, 1, num_cores_per_chip][i] for i in axes]
    else:
        host_bounds = [[2, 2, 1, num_cores_per_chip][i] for i in axes]
    host_sizes = [1]
    for host_bound in host_bounds:
        host_sizes.append(host_sizes[-1] * host_bound)
    host_size = host_sizes.pop()
    if ring_size < host_size:
        assert not can_split_host_across_rings
        raise ValueError('Rings too small for can_split_host_across_rings = False: %d' % ring_size)
    ring_bounds = [ring_bounds[i] for i in axes]
    if ring_bounds < host_bounds:
        raise ValueError('ring_bounds %s should be >= host_bounds %s' % (ring_bounds, host_bounds))
    ring_sizes = [1]
    for ring_bound in ring_bounds:
        ring_sizes.append(ring_sizes[-1] * ring_bound)
    ring_sizes.pop()
    cores = _enumerate_cores(bounds, ring_bounds, ring_sizes, host_bounds, host_sizes)
    core_locations = []
    for core in cores:
        core = [core[axes.index(i)] for i in range(4)]
        core_locations.append(_CoreLocation(core[0], core[1], core[2], core[3]))
    return core_locations