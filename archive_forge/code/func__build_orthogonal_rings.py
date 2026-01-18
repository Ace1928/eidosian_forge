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
def _build_orthogonal_rings(core_locations: List[_CoreLocation], ring_size: int, rotate_ring_across_rings: bool) -> List[_CoreLocation]:
    """Build two all-reduce rings orthogonal to each other.

  One ring includes every `ring_size` consecutive core locations. It is usually
  applied to the model-parallel dimension of a mesh to achieve best 1D
  all-reduce performance. The other ring includes core locations separated by
  a stride of `ring_size`. It is usually applied to the data-parallel dimension
  of a mesh to get predictable strided all-reduce performance.

  Args:
    core_locations: A list of core locations expressed as [x, y, z, core].
    ring_size: The number of core locations in the consecutive ring.
    rotate_ring_across_rings: Build column-major secondary rings.

  Returns:
    A permutation of the input list forming the described rings.
  """
    num_cores = len(core_locations)
    permutation = _build_all_reduce_ring(core_locations[:ring_size])
    for r in range(0, num_cores, ring_size):
        core_locations[r:r + ring_size] = [core_locations[r + permutation[i]] for i in range(ring_size)]
    logging.vlog(1, 'Permutated core locations: %s', core_locations)
    transposed = []
    for i in range(ring_size):
        transposed += [core_locations[g + i] for g in range(0, num_cores, ring_size)]
    num_rings = int(num_cores / ring_size)
    permutation = _build_all_reduce_ring(transposed[:num_rings], rotate=rotate_ring_across_rings)
    for r in range(0, num_cores, num_rings):
        transposed[r:r + num_rings] = [transposed[r + permutation[i]] for i in range(num_rings)]
    untransposed = []
    for i in range(num_rings):
        untransposed += [transposed[g + i] for g in range(0, num_cores, num_rings)]
    logging.vlog(1, 'Stride-permutated core locations: %s', untransposed)
    return untransposed