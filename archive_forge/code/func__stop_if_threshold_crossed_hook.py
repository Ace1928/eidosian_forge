import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _stop_if_threshold_crossed_hook(estimator, metric_name, threshold, higher_is_better, eval_dir, min_steps, run_every_secs, run_every_steps):
    """Creates early-stopping hook to stop training if threshold is crossed."""
    if eval_dir is None:
        eval_dir = estimator.eval_dir()
    is_lhs_better = operator.gt if higher_is_better else operator.lt
    greater_or_lesser = 'greater than' if higher_is_better else 'less than'

    def stop_if_threshold_crossed_fn():
        """Returns `True` if the given metric crosses specified threshold."""
        eval_results = read_eval_metrics(eval_dir)
        for step, metrics in eval_results.items():
            if step < min_steps:
                continue
            val = metrics[metric_name]
            if is_lhs_better(val, threshold):
                tf.compat.v1.logging.info('At step %s, metric "%s" has value %s which is %s the configured threshold (%s) for early stopping.', step, metric_name, val, greater_or_lesser, threshold)
                return True
        return False
    return make_early_stopping_hook(estimator=estimator, should_stop_fn=stop_if_threshold_crossed_fn, run_every_secs=run_every_secs, run_every_steps=run_every_steps)