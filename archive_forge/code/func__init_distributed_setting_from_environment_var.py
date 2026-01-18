from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _init_distributed_setting_from_environment_var(self, tf_config):
    """Initialize distributed properties based on `tf_config`."""
    self._service = _validate_service(tf_config.get(_SERVICE_KEY))
    self._cluster_spec = tf.train.ClusterSpec(tf_config.get(_CLUSTER_KEY, {}))
    task_env = tf_config.get(_TASK_ENV_KEY, {})
    if self._cluster_spec and TaskType.MASTER in self._cluster_spec.jobs:
        return self._init_distributed_setting_from_environment_var_with_master(tf_config)
    if self._cluster_spec:
        self._task_type, self._task_id = _validate_task_type_and_task_id(self._cluster_spec, task_env, TaskType.CHIEF)
        self._evaluation_master = _get_eval_session_master(self._task_type, tf_config)
        if self._task_type != TaskType.EVALUATOR:
            self._master = _get_session_master(self._cluster_spec, self._task_type, self._task_id, tf_config)
            self._num_ps_replicas = _count_ps(self._cluster_spec)
            self._num_worker_replicas = _count_worker(self._cluster_spec, chief_task_type=TaskType.CHIEF)
            self._global_id_in_cluster = _get_global_id_in_cluster(self._cluster_spec, self._task_type, self._task_id, chief_task_type=TaskType.CHIEF)
        else:
            self._cluster_spec = tf.train.ClusterSpec({})
            self._master = _LOCAL_MASTER
            self._num_ps_replicas = 0
            self._num_worker_replicas = 0
            self._global_id_in_cluster = None
        self._is_chief = self._task_type == TaskType.CHIEF
    else:
        self._task_type = task_env.get(_TASK_TYPE_KEY, TaskType.WORKER)
        self._task_id = int(task_env.get(_TASK_ID_KEY, 0))
        self._global_id_in_cluster = 0
        if self._task_type != TaskType.WORKER:
            raise ValueError('If "cluster" is not set in TF_CONFIG, task type must be WORKER.')
        if self._task_id != 0:
            raise ValueError('If "cluster" is not set in TF_CONFIG, task index must be 0.')
        self._master = tf_config.get(_SESSION_MASTER_KEY, _LOCAL_MASTER)
        self._evaluation_master = tf_config.get(_EVAL_SESSION_MASTER_KEY, _LOCAL_MASTER)
        self._is_chief = True
        self._num_ps_replicas = 0
        self._num_worker_replicas = 1