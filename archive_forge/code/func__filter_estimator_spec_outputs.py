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
def _filter_estimator_spec_outputs(spec):
    """Filters tensors and ops from an EstimatorSpec and returns a dictionary."""
    if spec.mode == ModeKeys.TRAIN:
        return dict(predictions=spec.predictions, train_op=spec.train_op)
    return dict(predictions=spec.predictions)