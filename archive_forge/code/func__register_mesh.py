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
def _register_mesh(self, mesh: layout_lib.Mesh):
    """Idempotently register `mesh` with the dtensor device."""
    with self._mesh_lock:
        if mesh not in self._meshes:
            _pywrap_dtensor_device.AddMesh(self._device_info, mesh.to_string(), False)
            self._meshes.add(mesh)
            if mesh.device_type().upper() == 'TPU':
                logging.info('Registering virtual 1:1 mapped host mesh %s for mesh %s', mesh.host_mesh().to_string(), mesh.to_string())
                _pywrap_dtensor_device.AddMesh(self._device_info, mesh.host_mesh().to_string(), True)
                self._meshes.add(mesh.host_mesh())