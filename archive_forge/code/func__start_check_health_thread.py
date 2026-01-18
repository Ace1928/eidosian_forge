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
def _start_check_health_thread(self):
    dummy_value = array_ops.identity([])
    logging.info('Waiting for the cluster, timeout = %s', self._check_health_initial_timeout or 'inf')
    try:
        self._host_cross_device_ops.reduce(reduce_util.ReduceOp.SUM, dummy_value, dummy_value, options=collective_util.Options(timeout_seconds=self._check_health_initial_timeout, implementation=collective_util.CommunicationImplementation.RING))
        if context.is_async():
            context.async_wait()
    except errors.DeadlineExceededError:
        raise RuntimeError('Timeout waiting for the cluster, timeout is %d seconds' % self._check_health_initial_timeout)
    logging.info('Cluster is ready.')
    self._check_health_thread_should_stop = threading.Event()
    self._check_health_thread = threading.Thread(target=self._check_health, daemon=True)
    self._check_health_thread.start()