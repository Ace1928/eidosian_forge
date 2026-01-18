import sys
import unittest
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import cluster_resolver
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import one_device_strategy as one_device_lib
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util as framework_test_util
from tensorflow.python.platform import flags
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export
def _create_tpu_strategy():
    FLAGS = flags.FLAGS
    global _did_connect_to_cluster
    global _topology
    try:
        resolver = tpu_cluster_resolver.TPUClusterResolver()
        did_automatically_resolve = True
    except ValueError:
        did_automatically_resolve = False
        resolver = tpu_cluster_resolver.TPUClusterResolver(tpu=hasattr(FLAGS, 'tpu') and FLAGS.tpu or '', zone=hasattr(FLAGS, 'zone') and FLAGS.zone or None, project=hasattr(FLAGS, 'project') and FLAGS.project or None)
    if not _did_connect_to_cluster:
        if getattr(FLAGS, 'tpu', '') or did_automatically_resolve:
            remote.connect_to_cluster(resolver)
            _did_connect_to_cluster = True
        _topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
    device_assignment = None
    if use_single_core:
        device_assignment = device_assignment_lib.DeviceAssignment(_topology, core_assignment=device_assignment_lib.SINGLE_CORE_ASSIGNMENT)
    if tf2.enabled():
        strategy = tpu_lib.TPUStrategyV2(resolver, device_assignment, experimental_spmd_xla_partitioning=enable_spmd_xla_paritioning, **kwargs)
    else:
        strategy = tpu_lib.TPUStrategyV1(resolver, steps_per_run, device_assignment, **kwargs)
    if enable_packed_variable and enable_spmd_xla_paritioning:
        raise ValueError('Packed Variable is not compatiable with SPMD mode')
    strategy._enable_packed_variable_in_eager_mode = enable_packed_variable
    return strategy