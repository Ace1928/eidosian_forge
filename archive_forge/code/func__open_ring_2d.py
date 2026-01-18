import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
def _open_ring_2d(x_size: int, y_size: int, z_coord: int) -> List[Tuple[int, int, int]]:
    """Ring-order of a X by Y mesh, with a fixed Z coordinate.

  For example, in a 4x4 mesh, this returns the following order.
    0 -- 1 -- 2 -- 3
    |    |    |    |
    15-- 6 -- 5 -- 4
    |    |    |    |
    14-- 7 -- 8 -- 9
    |    |    |    |
    13-- 12-- 11-- 10

  Note that chip 0 is not included in the output.

  Args:
    x_size: An integer represents the mesh size in the x-dimension. Must be
      larger than 1.
    y_size: An integer represents the mesh size in the y-dimension. Must be
      larger than 1.
    z_coord: An integer represents the z-coordinate to use for the chips in the
      ring.

  Returns:
    A list of (x,y,z) triples in ring order.
  """
    ret = []
    for i in range(y_size // 2):
        for j in range(1, x_size):
            ret.append((j, 2 * i, z_coord))
        for j in range(x_size - 1, 0, -1):
            ret.append((j, 2 * i + 1, z_coord))
    for i in range(y_size - 1, 0, -1):
        ret.append((0, i, z_coord))
    return ret