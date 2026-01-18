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
@tf_export(v1=['distribute.experimental.MultiWorkerMirroredStrategy'])
class CollectiveAllReduceStrategyV1(distribute_lib.StrategyV1):
    __doc__ = CollectiveAllReduceStrategy.__doc__
    _collective_key_base = 0

    def __init__(self, communication=collective_util.CommunicationImplementation.AUTO, cluster_resolver=None):
        """Initializes the object."""
        communication_options = collective_util.Options(implementation=communication)
        super(CollectiveAllReduceStrategyV1, self).__init__(CollectiveAllReduceExtended(self, cluster_resolver=cluster_resolver, communication_options=communication_options))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('MultiWorkerMirroredStrategy')
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_workers').set(self.extended._num_workers)
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_gpu_per_worker').set(self.extended._num_devices_per_worker if self.extended._local_device_type == 'GPU' else 0)