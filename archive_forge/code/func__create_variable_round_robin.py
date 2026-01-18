import functools
import os
import threading
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as base_cluster_resolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import server_lib
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def _create_variable_round_robin(self, next_creator, **kwargs):
    with ops.colocate_with(None, ignore_existing=True):
        with ops.device('/job:ps/task:%d/device:CPU:0' % (self._variable_count % self._num_ps)):
            var = next_creator(**kwargs)
            log_method = logging.info if os.getenv('TF_PSS_VERBOSE_VARIABLE_PLACEMENT') else logging.debug
            log_method('Creating variable (name:%s, shape:%r) on /job:ps/task:%d/device:CPU:0', var.name, var.shape, self._variable_count % self._num_ps)
            self._variable_count += 1
            return var