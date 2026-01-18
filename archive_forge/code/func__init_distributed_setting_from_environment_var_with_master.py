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
def _init_distributed_setting_from_environment_var_with_master(self, tf_config):
    """Initialize distributed properties for legacy cluster with `master`."""
    if TaskType.CHIEF in self._cluster_spec.jobs:
        raise ValueError('If `master` node exists in `cluster`, job `chief` is not supported.')
    task_env = tf_config.get(_TASK_ENV_KEY, {})
    self._task_type, self._task_id = _validate_task_type_and_task_id(self._cluster_spec, task_env, TaskType.MASTER)
    if self._task_type == TaskType.EVALUATOR:
        raise ValueError('If `master` node exists in `cluster`, task_type `evaluator` is not supported.')
    self._global_id_in_cluster = _get_global_id_in_cluster(self._cluster_spec, self._task_type, self._task_id, chief_task_type=TaskType.MASTER)
    self._master = _get_session_master(self._cluster_spec, self._task_type, self._task_id, tf_config)
    self._evaluation_master = _get_eval_session_master(self._task_type, tf_config)
    self._num_ps_replicas = _count_ps(self._cluster_spec)
    self._num_worker_replicas = _count_worker(self._cluster_spec, chief_task_type=TaskType.MASTER)
    self._is_chief = self._task_type == TaskType.MASTER