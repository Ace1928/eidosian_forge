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
@estimator_export(v1=['estimator.tpu.InputPipelineConfig'])
class InputPipelineConfig(object):
    """Please see the definition of these values in TPUConfig.

  @compatibility(TF2)
  TPU Estimator manages its own TensorFlow graph and session, so it is not
  compatible with TF2 behaviors. We recommend that you migrate to the newer
  `tf.distribute.TPUStrategy`. See the
  [TPU guide](https://www.tensorflow.org/guide/tpu) for details.
  @end_compatibility
  """
    PER_SHARD_V1 = 1
    PER_HOST_V1 = 2
    PER_HOST_V2 = 3
    BROADCAST = 4
    SLICED = 5