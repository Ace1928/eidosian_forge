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
def _canonicalize_receiver_tensors(receiver_tensors):
    """Converts receiver tensors to the expected format of `as_signature_def`."""
    for tensor in tf.nest.flatten(receiver_tensors):
        if not isinstance(tensor, tf.Tensor):
            raise ValueError('All receiver tensors must be tensors (composite tensors are not yet supported).')
    if isinstance(receiver_tensors, dict):
        return receiver_tensors
    return {export_utils.SINGLE_RECEIVER_DEFAULT_NAME: receiver_tensors}