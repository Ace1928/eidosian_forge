import copy
import threading
import time
import weakref
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def _update_config_proto(self, config_proto):
    updated_config = copy.deepcopy(config_proto)
    rewrite_options = updated_config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')
    if not ops.executing_eagerly_outside_functions() and self._communication_options.implementation == collective_util.CommunicationImplementation.NCCL:
        updated_config.experimental.collective_nccl = True
    if not self._cluster_spec:
        return updated_config
    assert self._task_type
    assert self._task_id is not None
    updated_config.experimental.collective_group_leader = multi_worker_util.collective_leader(self._cluster_spec, self._task_type, self._task_id)
    del updated_config.device_filters[:]
    updated_config.device_filters.append('/job:%s/task:%d' % (self._task_type, self._task_id))
    return updated_config