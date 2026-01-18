from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import time
import six
import tensorflow as tf
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import server_lib
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import exporter as exporter_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _start_std_server(self, config):
    """Creates, starts, and returns a server_lib.Server."""
    if not config.cluster_spec or not config.task_type or config.task_id is None:
        raise RuntimeError('Could not start server; be sure to specify cluster_spec, task_type, and task in RunConfig or set the TF_CONFIG environment variable.')
    if not config.master:
        jobs = config.cluster_spec.jobs
        if len(jobs) == 1 and len(config.cluster_spec.job_tasks(jobs[0])) == 1 and (config.task_type in _TRAINER_JOBS):
            tf.compat.v1.logging.info('Skip starting Tensorflow server as there is only one node in the cluster.')
            return
        else:
            raise RuntimeError('Could not start server; be sure to specify master in RunConfig or set the TF_CONFIG environment variable.')
    tf.compat.v1.logging.info('Start Tensorflow server.')
    if config.session_config is None:
        session_config = tf.compat.v1.ConfigProto(log_device_placement=False)
    else:
        session_config = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options=config.session_config.gpu_options)
    server = server_lib.Server(config.cluster_spec, job_name=config.task_type, task_index=config.task_id, config=session_config, start=False, protocol=config.protocol)
    server.start()
    return server