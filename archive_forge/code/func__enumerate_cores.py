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
def _enumerate_cores(bounds: List[int], ring_bounds: List[int], ring_sizes: List[int], host_bounds: List[int], host_sizes: List[int]) -> List[List[int]]:
    """Enumerates cores within `bounds` from fatest to slowest varying axes.

  Args:
    bounds: Upper bounds of axes, from fastest to slowest varying.
    ring_bounds: Upper bounds of ring size per axis in the same axis order.
    ring_sizes: Number consecutive cores in the ring built so far, cumulatively.
    host_bounds: Number of axis values per host in the same axis order.
    host_sizes: Number consecutive cores on one host, cumulatively.

  Returns:
    Cores represented as a list of 4 integers in the same axis order.
  """
    if not bounds:
        return [[]]
    partials = _enumerate_cores(bounds[:-1], ring_bounds[:-1], ring_sizes[:-1], host_bounds[:-1], host_sizes[:-1])
    results = []
    for ring_i in range(0, bounds[-1], ring_bounds[-1]):
        for ring_j in range(0, len(partials), ring_sizes[-1]):
            for host_i in range(ring_i, ring_i + ring_bounds[-1], host_bounds[-1]):
                for host_j in range(ring_j, ring_j + ring_sizes[-1], host_sizes[-1]):
                    for i in range(host_i, host_i + host_bounds[-1]):
                        for j in range(host_j, host_j + host_sizes[-1]):
                            results.append(partials[j] + [i])
    return results