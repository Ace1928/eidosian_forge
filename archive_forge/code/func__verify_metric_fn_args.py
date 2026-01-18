from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _verify_metric_fn_args(metric_fn):
    args = set(function_utils.fn_args(metric_fn))
    invalid_args = list(args - _VALID_METRIC_FN_ARGS)
    if invalid_args:
        raise ValueError('metric_fn (%s) has following not expected args: %s' % (metric_fn, invalid_args))