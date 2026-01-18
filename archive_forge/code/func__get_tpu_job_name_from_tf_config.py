from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.tpu import util as util_lib
def _get_tpu_job_name_from_tf_config():
    """Extracts the TPU job name from TF_CONFIG env variable."""
    tf_config = json.loads(os.environ.get(_TF_CONFIG_ENV, '{}'))
    tpu_job_name = tf_config.get(_SERVICE_KEY, {}).get(_TPU_WORKER_JOB_NAME)
    if tpu_job_name:
        tf.compat.v1.logging.info('Load TPU job name from TF_CONFIG: %s', tpu_job_name)
    return tpu_job_name