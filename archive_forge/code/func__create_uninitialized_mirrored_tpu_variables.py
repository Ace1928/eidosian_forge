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
def _create_uninitialized_mirrored_tpu_variables(**kwargs):
    """Returns a list of `tf.Variable`s.

      The list contains `number_replicas` `tf.Variable`s and can be used to
      initialize a `TPUMirroredVariable`.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
    if kwargs.get('initial_value', None) is None:
        return _create_mirrored_tpu_variables(**kwargs)
    value_list = []
    for i, d in enumerate(devices):
        with ops.device(d):
            if i == 0:
                initial_value = kwargs.get('initial_value', None)
                with maybe_init_scope():
                    if initial_value is not None:
                        if callable(initial_value):
                            initial_value = initial_value()
                        initial_value = ops.convert_to_tensor(initial_value, dtype=kwargs.get('dtype', None))
            if i > 0:
                var0name = value_list[0].name.split(':')[0]
                kwargs['name'] = '%s/replica_%d/' % (var0name, i)
            kwargs['initial_value'] = initial_value
            if kwargs.get('dtype', None) is None:
                kwargs['dtype'] = kwargs['initial_value'].dtype
            if kwargs.get('shape', None) is None:
                kwargs['shape'] = kwargs['initial_value'].shape
            with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                v = uninitialized_variable_creator(**kwargs)
            assert not isinstance(v, tpu_values.TPUMirroredVariable)
            value_list.append(v)
    return value_list