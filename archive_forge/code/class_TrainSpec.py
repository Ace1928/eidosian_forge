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
@estimator_export('estimator.TrainSpec')
class TrainSpec(collections.namedtuple('TrainSpec', ['input_fn', 'max_steps', 'hooks', 'saving_listeners'])):
    """Configuration for the "train" part for the `train_and_evaluate` call.

  `TrainSpec` determines the input data for the training, as well as the
  duration. Optional hooks run at various stages of training.

  Usage:

  >>> train_spec = tf.estimator.TrainSpec(
  ...    input_fn=lambda: 1,
  ...    max_steps=100,
  ...    hooks=[_StopAtSecsHook(stop_after_secs=10)],
  ...    saving_listeners=[_NewCheckpointListenerForEvaluate(None, 20, None)])
  >>> train_spec.saving_listeners[0]._eval_throttle_secs
  20
  >>> train_spec.hooks[0]._stop_after_secs
  10
  >>> train_spec.max_steps
  100
  """

    def __new__(cls, input_fn, max_steps=None, hooks=None, saving_listeners=None):
        """Creates a validated `TrainSpec` instance.

    Args:
      input_fn: A function that provides input data for training as minibatches.
        See [Premade Estimators](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
          for more information. The function should construct and return one of
        the following:
          * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
          * A tuple (features, labels): Where features is a `Tensor` or a
            dictionary of string feature name to `Tensor` and labels is a
            `Tensor` or a dictionary of string label name to `Tensor`.
      max_steps: Int. Positive number of total steps for which to train model.
        If `None`, train forever. The training `input_fn` is not expected to
        generate `OutOfRangeError` or `StopIteration` exceptions. See the
        `train_and_evaluate` stop condition section for details.
      hooks: Iterable of `tf.train.SessionRunHook` objects to run on all workers
        (including chief) during training.
      saving_listeners: Iterable of `tf.estimator.CheckpointSaverListener`
        objects to run on chief during training.

    Returns:
      A validated `TrainSpec` object.

    Raises:
      ValueError: If any of the input arguments is invalid.
      TypeError: If any of the arguments is not of the expected type.
    """
        _validate_input_fn(input_fn)
        if max_steps is not None and max_steps <= 0:
            raise ValueError('Must specify max_steps > 0, given: {}'.format(max_steps))
        hooks = _validate_hooks(hooks)
        saving_listeners = _validate_saving_listeners(saving_listeners)
        return super(TrainSpec, cls).__new__(cls, input_fn=input_fn, max_steps=max_steps, hooks=hooks, saving_listeners=saving_listeners)