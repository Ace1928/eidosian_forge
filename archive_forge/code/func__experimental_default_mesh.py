import contextlib
import logging
import threading
from typing import Any, List, Sequence, Set
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import _pywrap_utils
@contextlib.contextmanager
def _experimental_default_mesh(self, mesh: layout_lib.Mesh):
    """Sets a default mesh for all ops in the scope.

    Note: This is an internal helper method, which is not user facing api.

    Useful for requesting a specific mesh for ops which would have no inferred
    layout, e.g. tf.zeros.

    Args:
      mesh: A Mesh to be used for ops without Mesh.

    Yields:
      Nothing.
    """
    previous_default = self._current_default_mesh
    self._register_mesh(mesh)
    _pywrap_dtensor_device.ExperimentalSetDefaultMesh(self._device_info, mesh.to_string().encode('utf-8'))
    self._current_default_mesh = mesh
    yield
    _pywrap_dtensor_device.ExperimentalClearDefaultMesh(self._device_info)
    if previous_default:
        _pywrap_dtensor_device.ExperimentalSetDefaultMesh(self._device_info, previous_default.to_string().encode('utf-8'))
    self._current_default_mesh = previous_default