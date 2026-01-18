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
def _get_model_dir(tf_config, model_dir):
    """Returns `model_dir` based user provided `tf_config` or `model_dir`."""
    if model_dir == '':
        raise ValueError('model_dir should be non-empty.')
    model_dir_in_tf_config = tf_config.get('model_dir')
    if model_dir_in_tf_config == '':
        raise ValueError('model_dir in TF_CONFIG should be non-empty.')
    if model_dir_in_tf_config:
        if model_dir and model_dir_in_tf_config != model_dir:
            raise ValueError('`model_dir` provided in RunConfig construct, if set, must have the same value as the model_dir in TF_CONFIG. model_dir: {}\nTF_CONFIG["model_dir"]: {}.\n'.format(model_dir, model_dir_in_tf_config))
        tf.compat.v1.logging.info('Using model_dir in TF_CONFIG: %s', model_dir_in_tf_config)
    return model_dir or model_dir_in_tf_config