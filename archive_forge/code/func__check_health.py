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
def _check_health(self):
    while True:
        if self._check_health_thread_should_stop.is_set():
            return
        for job in self._cluster_spec.jobs:
            for task_id in range(self._cluster_spec.num_tasks(job)):
                peer = '/job:{}/replica:0/task:{}'.format(job, task_id)
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        context.context().check_collective_ops_peer_health(peer, timeout_in_ms=self._check_health_timeout * 1000)
                        break
                    except (errors.UnavailableError, errors.FailedPreconditionError, errors.DeadlineExceededError) as e:
                        if attempts < self._check_health_retry_limit:
                            logging.warning('%s seems down, retrying %d/%d', peer, attempts, self._check_health_retry_limit)
                            continue
                        logging.error('Cluster check alive failed, %s is down, aborting collectives: %s', peer, e)
                        context.context().abort_collective_ops(errors.UNAVAILABLE, 'cluster check alive failed, {} is down'.format(peer))
                        return
                    except Exception as e:
                        logging.error('Unexpected exception in check alive: %s', e)
                        context.context().abort_collective_ops(errors.INTERNAL, 'unexecpted exception in check alive: %s' % e)
                        return
        time.sleep(self._check_health_interval)