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
def _create_tpu_topology(core_locations: List[_CoreLocation], num_tasks: int, num_devices_per_task: int) -> topology.Topology:
    """Returns a Topology object build from a _CoreLocation list.

  Args:
    core_locations: A list of _CoreLocation objects sorted first by TF task ID
      and then by per-task device ordinals.
    num_tasks: The number of TF tasks in the cluster.
    num_devices_per_task: The number of TPU devices local to each task.
  """
    assert min([l.x for l in core_locations]) == 0
    assert min([l.y for l in core_locations]) == 0
    assert min([l.z for l in core_locations]) == 0
    assert min([l.core for l in core_locations]) == 0
    x_max = max([l.x for l in core_locations])
    y_max = max([l.y for l in core_locations])
    z_max = max([l.z for l in core_locations])
    core_max = max([l.core for l in core_locations])
    mesh_shape = [x_max + 1, y_max + 1, z_max + 1, core_max + 1]
    device_coordinates = [[l.x, l.y, l.z, l.core] for l in core_locations]
    device_coordinates = np.asarray(device_coordinates).reshape(num_tasks, num_devices_per_task, 4)
    return topology.Topology(mesh_shape=mesh_shape, device_coordinates=device_coordinates)