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
def _initialize_multi_worker(self, cluster_resolver):
    """Initializes the object for multi-worker training."""
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_resolver.cluster_spec())
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if task_type is None or task_id is None:
        raise ValueError('When `cluster_spec` is given, you must also specify `task_type` and `task_id`.')
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._id_in_cluster = multi_worker_util.id_in_cluster(self._cluster_spec, self._task_type, self._task_id)
    self._num_workers = multi_worker_util.worker_count(cluster_spec, task_type)
    if not self._num_workers:
        raise ValueError('No `worker`, `chief` or `evaluator` tasks can be found in `cluster_spec`.')
    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type, task_id)
    self._worker_device = '/job:%s/task:%d' % (task_type, task_id)
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)
    if ops.executing_eagerly_outside_functions() and (not getattr(self, '_local_or_standalone_client_mode', False)):
        context.context().configure_collective_ops(collective_leader=multi_worker_util.collective_leader(cluster_spec, task_type, task_id), scoped_allocator_enabled_ops=('CollectiveReduce',), device_filters=('/job:%s/task:%d' % (task_type, task_id),))
        self._collective_ops_configured = True
        if context.context().coordination_service is None:
            coordinated_jobs = ['chief', 'worker']
            if task_type in coordinated_jobs:
                coordinated_job_config = []
                for job in coordinated_jobs:
                    if job in cluster_spec.jobs:
                        coordinated_job_config.append(coordination_config_pb2.CoordinatedJob(name=job, num_tasks=cluster_spec.num_tasks(job)))
                context.context().configure_coordination_service(service_type='standalone', service_leader=multi_worker_util.coordination_leader(cluster_spec), coordinated_jobs=coordinated_job_config)
    if context.executing_eagerly() and (not getattr(self, '_std_server_started', False)) and (not getattr(self, '_local_or_standalone_client_mode', False)):
        config_proto = copy.deepcopy(context.context().config)
        config_proto = self._update_config_proto(config_proto)
        if config_proto.experimental.coordination_config.service_type:
            self._enable_check_health = False
        if hasattr(cluster_resolver, 'port'):
            port = cluster_resolver.port
        else:
            port = 0
        server_def = tensorflow_server_pb2.ServerDef(cluster=cluster_spec.as_cluster_def(), default_session_config=config_proto, job_name=task_type, task_index=task_id, protocol=cluster_resolver.rpc_layer or 'grpc', port=port)
        context.context().enable_collective_ops(server_def)
        self._std_server_started = True
        context.context().ensure_initialized()
        logging.info('Enabled multi-worker collective ops with available devices: %r', context.context().devices())
    local_devices, local_device_type = self._initialize_local_devices(cluster_resolver, self._worker_device)
    if local_device_type == 'TPU':
        tpu_cluster_resolver.initialize_tpu_system()
    self._collective_keys = cross_device_utils.CollectiveKeys(group_key_start=1 + self._collective_key_base)
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(devices=local_devices, group_size=len(local_devices) * self._num_workers, options=self._communication_options, collective_keys=self._collective_keys)
    self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(devices=[self._worker_device], group_size=self._num_workers, options=self._communication_options, collective_keys=self._collective_keys)
    super(CollectiveAllReduceExtended, self)._initialize_single_worker(local_devices)
    self._default_device = '/job:%s/task:%d' % (task_type, task_id)
    self._num_devices_per_worker = len(local_devices)
    self._local_device_type = local_device_type
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()
    if self._enable_check_health and context.executing_eagerly():
        self._start_check_health_thread()
    else:
        logging.info('Check health not enabled.')
    logging.info('MultiWorkerMirroredStrategy with cluster_spec = %r, task_type = %r, task_id = %r, num_workers = %r, local_devices = %r, communication = %s', cluster_spec.as_dict(), task_type, task_id, self._num_workers, local_devices, self._communication_options.implementation)