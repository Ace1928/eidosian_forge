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