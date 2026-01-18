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
def _create_multi_worker_mirrored():
    tf_config = cluster_resolver.TFConfigClusterResolver()
    master = tf_config.master()
    if tf_config.rpc_layer:
        master = master[len('%s://' % tf_config.rpc_layer):]
    resolver = cluster_resolver.SimpleClusterResolver(cluster_spec=tf_config.cluster_spec(), task_type=tf_config.task_type, task_id=tf_config.task_id, master=master, environment=tf_config.environment, num_accelerators={'GPU': required_gpus}, rpc_layer=tf_config.rpc_layer or 'grpc')
    CollectiveAllReduceExtended._enable_check_health = False
    context.context().configure_coordination_service(service_type='')
    with context.eager_mode():
        strategy = CollectiveAllReduceStrategy(cluster_resolver=resolver)
    if not use_merge_call:
        strategy.extended._use_merge_call = lambda: False
    try:
        multi_process_runner.get_barrier().wait()
    except ValueError:
        pass
    return strategy