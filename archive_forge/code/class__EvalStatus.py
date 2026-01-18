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
class _EvalStatus(object):
    """The status of an evaluation event.

  For local training and evaluation, the status can only be `EVALUATED` as
  `Estimator.train` always generates a new checkpoint.

  For distributed training and evaluation, a separated evaluator keeps looking
  for new checkpoint. So, multiple situations might occur:

  - EVALUATED: A new checkpoint is found since last evaluation.
      `Estimator.evaluate` will be invoked.
  - MISSING_CHECKPOINT: No checkpoint can be found. Typically, this means
      the trainer has not yet produced any checkpoint.
  - NO_NEW_CHECKPOINT: No new checkpoint can be found since last evaluation.
      Typically, this means the trainer has not yet produced any new checkpoint.
  """
    EVALUATED = 'evaluated'
    MISSING_CHECKPOINT = 'missing checkpoint'
    NO_NEW_CHECKPOINT = 'no new checkpoint'