from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import gc
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _verify_compare_fn_args(compare_fn):
    """Verifies compare_fn arguments."""
    args = set(util.fn_args(compare_fn))
    if 'best_eval_result' not in args:
        raise ValueError('compare_fn (%s) must include best_eval_result argument.' % compare_fn)
    if 'current_eval_result' not in args:
        raise ValueError('compare_fn (%s) must include current_eval_result argument.' % compare_fn)
    non_valid_args = list(args - set(['best_eval_result', 'current_eval_result']))
    if non_valid_args:
        raise ValueError('compare_fn (%s) has following not expected args: %s' % (compare_fn, non_valid_args))