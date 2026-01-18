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