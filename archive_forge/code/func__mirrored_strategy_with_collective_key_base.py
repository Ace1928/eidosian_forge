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
def _mirrored_strategy_with_collective_key_base(devices):
    required_cpus_nums = sum((1 for d in devices if tf_device.DeviceSpec.from_string(d).device_type == 'CPU'))
    if required_cpus_nums > len(context.context().list_logical_devices('CPU')):
        context._reset_context()
        test_util.set_logical_devices_to_at_least('CPU', required_cpus_nums)
    mirrored_lib.MirroredStrategyV1._collective_key_base += 100000
    mirrored_lib.MirroredStrategy._collective_key_base += 100000
    return MirroredStrategy(devices)