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
def _connect_to_cluster(self, coordinator_name: str):
    if coordinator_name in ['worker', 'ps']:
        raise ValueError("coordinator name should not be 'worker' or 'ps'.")
    cluster_spec = self._cluster_resolver.cluster_spec()
    self._num_workers = len(cluster_spec.as_dict().get('worker', ()))
    self._num_ps = len(cluster_spec.as_dict().get('ps', ()))
    device_filters = server_lib.ClusterDeviceFilters()
    for i in range(self._num_workers):
        device_filters.set_device_filters('worker', i, ['/job:ps', '/job:%s' % coordinator_name])
    for i in range(self._num_ps):
        device_filters.set_device_filters('ps', i, ['/job:worker', '/job:%s' % coordinator_name])
    os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'False'
    os.environ['TF_PS_DISABLE_ASYNC_EXECUTOR_GLOBALLY'] = 'True'
    logging.info('%s is now connecting to cluster with cluster_spec: %r', self.__class__.__name__, cluster_spec)
    remote.connect_to_cluster(cluster_spec, job_name=coordinator_name, protocol=self._cluster_resolver.rpc_layer, cluster_device_filters=device_filters)
    distribute_lib.distribution_strategy_replica_gauge.get_cell('ps_strategy_num_workers').set(self._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell('ps_strategy_num_ps').set(self._num_ps)