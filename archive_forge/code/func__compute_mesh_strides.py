import collections
import functools
import itertools
from typing import List, Dict, Optional, Union
import numpy as np
from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util.tf_export import tf_export
def _compute_mesh_strides(shape: List[int]) -> List[int]:
    strides = [1]
    for idx, dim_size in enumerate(reversed(shape[1:])):
        strides.append(strides[idx] * dim_size)
    strides.reverse()
    return strides