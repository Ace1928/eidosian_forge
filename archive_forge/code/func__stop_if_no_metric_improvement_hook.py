import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _stop_if_no_metric_improvement_hook(estimator, metric_name, max_steps_without_improvement, higher_is_better, eval_dir, min_steps, run_every_secs, run_every_steps):
    """Returns hook to stop training if given metric shows no improvement."""
    if eval_dir is None:
        eval_dir = estimator.eval_dir()
    is_lhs_better = operator.gt if higher_is_better else operator.lt
    increase_or_decrease = 'increase' if higher_is_better else 'decrease'

    def stop_if_no_metric_improvement_fn():
        """Returns `True` if metric does not improve within max steps."""
        eval_results = read_eval_metrics(eval_dir)
        best_val = None
        best_val_step = None
        for step, metrics in eval_results.items():
            if step < min_steps:
                continue
            val = metrics[metric_name]
            if best_val is None or is_lhs_better(val, best_val):
                best_val = val
                best_val_step = step
            if step - best_val_step >= max_steps_without_improvement:
                tf.compat.v1.logging.info('No %s in metric "%s" for %s steps, which is greater than or equal to max steps (%s) configured for early stopping.', increase_or_decrease, metric_name, step - best_val_step, max_steps_without_improvement)
                return True
        return False
    return make_early_stopping_hook(estimator=estimator, should_stop_fn=stop_if_no_metric_improvement_fn, run_every_secs=run_every_secs, run_every_steps=run_every_steps)