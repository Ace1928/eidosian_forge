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
class _EvalResult(collections.namedtuple('EvalResult', ['status', 'metrics', 'checkpoint_path'])):
    """_EvalResult holds the result of an evaluation event."""

    def __new__(cls, status, metrics=None, checkpoint_path=None):
        """Creates a validated `_EvalResult`.

    Args:
      status: See `_EvalStatus`.
      metrics: The evaluation results returned by `Estimator.evaluate`. Only set
        if status is `EVALUATED`.
      checkpoint_path: The corresponding checkpoint path for the `metrics`. Only
        set if status is `EVALUATED`.

    Returns:
      A validated `_EvalResult` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """
        if status != _EvalStatus.EVALUATED:
            if metrics:
                raise ValueError('metrics must be `None` if status is not {}; got status {}, metrics {}'.format(_EvalStatus.EVALUATED, status, metrics))
            if checkpoint_path:
                raise ValueError('checkpoint must be `None` if status is not {}; got status {}, checkpoint_path {}'.format(_EvalStatus.EVALUATED, status, checkpoint_path))
            return super(_EvalResult, cls).__new__(cls, status, metrics, checkpoint_path)
        assert status == _EvalStatus.EVALUATED
        if not metrics:
            raise ValueError('Internal error: `Estimator.evaluate` should never return empty metrics.')
        if not isinstance(metrics, dict):
            raise TypeError('`Estimator.evaluate` should return dict. Given {}.'.format(type(metrics)))
        if tf.compat.v1.GraphKeys.GLOBAL_STEP not in metrics:
            raise ValueError('Internal error: `Estimator.evaluate` result should have `global_step` in result. Given {}'.format(metrics))
        if not checkpoint_path:
            raise ValueError('Internal error: `checkpoint_path` should never be empty.')
        return super(_EvalResult, cls).__new__(cls, status, metrics, checkpoint_path)