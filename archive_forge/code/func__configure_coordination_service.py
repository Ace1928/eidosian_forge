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
def _configure_coordination_service(self, cluster_spec: base_cluster_resolver.ClusterSpec):
    if context.context().coordination_service is None:
        coordinated_jobs = ['worker', 'ps']
        coordinated_job_config = []
        for job in coordinated_jobs:
            if job in cluster_spec.jobs:
                coordinated_job_config.append(coordination_config_pb2.CoordinatedJob(name=job, num_tasks=cluster_spec.num_tasks(job)))
        context.context().configure_coordination_service(service_type='standalone', service_leader=multi_worker_util.coordination_leader(cluster_spec), heartbeat_timeout_in_ms=_HEARTBEAT_TIMEOUT_SECS * 1000, allow_new_incarnation_to_reconnect=True)