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
def _initialize_local_devices(self, cluster_resolver, worker_device):
    if isinstance(cluster_resolver, tfconfig_cluster_resolver.TFConfigClusterResolver):
        num_gpus = context.num_gpus()
        num_tpus = 0
    else:
        num_gpus = cluster_resolver.num_accelerators().get('GPU', 0)
        num_tpus = cluster_resolver.num_accelerators().get('TPU', 0)
    if num_gpus:
        local_device_type = 'GPU'
        num_local_devices = num_gpus
    elif num_tpus:
        local_device_type = 'TPU'
        num_local_devices = num_tpus
    else:
        local_device_type = 'CPU'
        num_local_devices = 1
    local_devices = tuple((f'{worker_device}/device:{local_device_type}:{i}' for i in range(num_local_devices)))
    return (local_devices, local_device_type)