from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import func_graph
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _wrap_and_verify_model_fn(model_fn, mode=None, config=None, params=None, input_signature=None):
    """Returns a function that only has only tensor arguments (features, labels).

  Args:
    model_fn: Model function. Must follow the signature defined in
      `tf.estimator.Estimator`.
    mode: Optional string `tf.estimstor.ModeKey`.
    config: Optional `estimator.RunConfig` object.
    params: Optional `dict` of hyperparameters.
    input_signature: Possibly nested TensorSpec of the tensor arguments.

  Returns:
    tuple of (
      function that only accepts tensor arguments (features and/or labels),
      whether the returned function expects a labels argument)
  """
    model_fn_lib.verify_model_fn_args(model_fn, params)
    args = function_utils.fn_args(model_fn)
    kwargs = {}
    if 'mode' in args:
        kwargs['mode'] = mode
    if 'params' in args:
        kwargs['params'] = params
    if 'config' in args:
        kwargs['config'] = config
    if 'labels' in args:
        if input_signature is None or len(input_signature) == 2:

            def wrapped_model_fn(features, labels=None):
                return model_fn(features=features, labels=labels, **kwargs)
        else:

            def wrapped_model_fn(features):
                return model_fn(features=features, labels=None, **kwargs)
    else:

        def wrapped_model_fn(features):
            return model_fn(features=features, **kwargs)
    return (wrapped_model_fn, 'labels' in args)