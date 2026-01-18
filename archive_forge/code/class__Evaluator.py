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
class _Evaluator(object):
    """A helper class to call `Estimator.evaluate` and export model."""

    def __init__(self, estimator, eval_spec, max_training_steps):
        self._estimator = estimator
        _assert_eval_spec(eval_spec)
        self._eval_spec = eval_spec
        self._is_final_export_triggered = False
        self._previous_ckpt_path = None
        self._last_warning_time = 0
        self._max_training_steps = max_training_steps

    @property
    def is_final_export_triggered(self):
        return self._is_final_export_triggered

    def evaluate_and_export(self):
        """Evaluate and (maybe) export the current model.

      Returns:
        A tuple of `EvalResult` instance and the export results.

      Raises:
        RuntimeError: for any unexpected internal error.
        TypeError: if evaluation result has wrong type.
      """
        latest_ckpt_path = self._estimator.latest_checkpoint()
        if not latest_ckpt_path:
            self._log_err_msg('Estimator is not trained yet. Will start an evaluation when a checkpoint is ready.')
            return (_EvalResult(status=_EvalStatus.MISSING_CHECKPOINT), [])
        if latest_ckpt_path == self._previous_ckpt_path:
            self._log_err_msg('No new checkpoint ready for evaluation. Skip the current evaluation pass as evaluation results are expected to be same for the same checkpoint.')
            return (_EvalResult(status=_EvalStatus.NO_NEW_CHECKPOINT), [])
        metrics = self._estimator.evaluate(input_fn=self._eval_spec.input_fn, steps=self._eval_spec.steps, name=self._eval_spec.name, checkpoint_path=latest_ckpt_path, hooks=self._eval_spec.hooks)
        eval_result = _EvalResult(status=_EvalStatus.EVALUATED, metrics=metrics, checkpoint_path=latest_ckpt_path)
        is_the_final_export = eval_result.metrics[tf.compat.v1.GraphKeys.GLOBAL_STEP] >= self._max_training_steps if self._max_training_steps else False
        export_results = self._export_eval_result(eval_result, is_the_final_export)
        if is_the_final_export:
            tf.compat.v1.logging.debug('Calling exporter with the `is_the_final_export=True`.')
            self._is_final_export_triggered = True
        self._last_warning_time = 0
        self._previous_ckpt_path = latest_ckpt_path
        return (eval_result, export_results)

    def _log_err_msg(self, message):
        """Prints warning `message` every 10 mins."""
        current_time = time.time()
        if current_time - self._last_warning_time > 600:
            tf.compat.v1.logging.warning(message)
            self._last_warning_time = current_time

    def _export_eval_result(self, eval_result, is_the_final_export):
        """Export `eval_result` according to exporters in `EvalSpec`."""
        export_dir_base = os.path.join(tf.compat.as_str_any(self._estimator.model_dir), tf.compat.as_str_any('export'))
        export_results = []
        for exporter in self._eval_spec.exporters:
            export_results.append(exporter.export(estimator=self._estimator, export_path=os.path.join(tf.compat.as_str_any(export_dir_base), tf.compat.as_str_any(exporter.name)), checkpoint_path=eval_result.checkpoint_path, eval_result=eval_result.metrics, is_the_final_export=is_the_final_export))
        return export_results