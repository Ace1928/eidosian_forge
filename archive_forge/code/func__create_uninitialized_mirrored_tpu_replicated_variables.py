import atexit
import collections
import contextlib
import copy
import functools
import weakref
from absl import logging
import numpy as np
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as tpu_cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import device_assignment as device_assignment_lib  # pylint: disable=unused-import
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_hardware_feature
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _create_uninitialized_mirrored_tpu_replicated_variables(**kwargs):
    """Returns a list of `TPUReplicatedVariable`s.

      The list consists of `num_replicas` `TPUReplicatedVariable`s and can be
      used to initialize a `TPUMirroredVariable`. Each `TPUReplicatedVariable`
      contains a list of `tf.Variable`s which are replicated to
      `num_cores_per_replica` logical cores to enable XLA SPMD compilation.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
    dtype = kwargs.get('dtype', None)
    shape = kwargs.get('shape', None)
    initial_value = kwargs.get('initial_value', None)
    if initial_value is None:
        return _create_mirrored_tpu_replicated_variables(**kwargs)
    with maybe_init_scope():
        if initial_value is not None:
            if callable(initial_value):
                initial_value = initial_value()
            initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
            kwargs['initial_value'] = initial_value
            if dtype is None:
                kwargs['dtype'] = kwargs['initial_value'].dtype
            if shape is None:
                kwargs['shape'] = kwargs['initial_value'].shape
    mirrored_replicated_var_list = []
    for replica_id in range(num_replicas):
        replicated_var_list = []
        for logic_core_id in range(num_cores_per_replica):
            with ops.device(self._tpu_devices[replica_id][logic_core_id]):
                v = uninitialized_variable_creator(**kwargs)
            replicated_var_list.append(v)
        replica_name = '{}/r:{}'.format(kwargs['name'], replica_id)
        tpu_replicated_var = tpu_replicated_variable.TPUReplicatedVariable(variables=replicated_var_list, name=replica_name)
        mirrored_replicated_var_list.append(tpu_replicated_var)
    return mirrored_replicated_var_list