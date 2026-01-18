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
def _build_all_reduce_ring(core_locations: List[_CoreLocation], rotate: bool=False) -> List[int]:
    """Reorders a list of TPU cores to optimize for AllReduce performance.

  This is ported from the C++ tensorflow::BuildAllReduceRing function,
  mixed with some logic from TF TPU's device_assignment._ring_3d.

  Args:
    core_locations: A list of core locations expressed as [x, y, z, core].
    rotate: If true, scan the cores in a column-major order. False by default.

  Returns:
    A permutation of the input list such that neighbors in the sequence are
    nearby in the TPU topology.
  """
    permutation = list(range(len(core_locations)))
    if not permutation:
        return permutation
    logging.vlog(2, 'Core locations in: %s', core_locations)
    first_column = min([l.x for l in core_locations])
    first_row = min([l.y for l in core_locations])
    same_z = len(set([l.z for l in core_locations])) == 1
    logging.vlog(2, 'first_column: %d', first_column)
    logging.vlog(2, 'first_row: %d', first_row)
    logging.vlog(2, 'same_z: %s', same_z)

    def _cmp_2d(ia: int, ib: int) -> int:
        if not rotate:
            a = core_locations[ia]
            b = core_locations[ib]
            a_first = a.x == first_column and a.y != first_row
            b_first = b.x == first_column and b.y != first_row
            if a_first != b_first:
                return -1 if b_first else 1
            if a.y != b.y:
                return b.y - a.y if a_first else a.y - b.y
            if a.x != b.x:
                return a.x - b.x if a.y % 2 == 0 else b.x - a.x
            return a.core - b.core
        else:
            a = core_locations[ia]
            b = core_locations[ib]
            a_first = a.y == first_row and a.x != first_column
            b_first = b.y == first_row and b.x != first_column
            if a_first != b_first:
                return -1 if b_first else 1
            if a.x != b.x:
                return b.x - a.x if a_first else a.x - b.x
            if a.y != b.y:
                return a.y - b.y if a.x % 2 == 0 else b.y - a.y
            return a.core - b.core

    def _cmp_3d(ia: int, ib: int) -> int:
        a = core_locations[ia]
        b = core_locations[ib]
        a_corner = a.x == first_column and a.y == first_row
        b_corner = b.x == first_column and b.y == first_row
        if a_corner and b_corner:
            return b.z - a.z if a.z != b.z else a.core - b.core
        if a_corner != b_corner:
            return -1 if b_corner else 1
        if a.z == b.z:
            return _cmp_2d(ia, ib) if a.z % 2 == 0 else -_cmp_2d(ia, ib)
        return a.z - b.z
    if same_z:
        permutation.sort(key=functools.cmp_to_key(_cmp_2d))
    else:
        permutation.sort(key=functools.cmp_to_key(_cmp_3d))
    logging.vlog(2, 'Permutation out: %s', permutation)
    return permutation