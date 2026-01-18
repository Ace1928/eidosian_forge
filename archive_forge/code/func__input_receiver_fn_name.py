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
def _input_receiver_fn_name(name):
    if name is None:
        return _RECEIVER_FN_NAME
    else:
        return '{}_{}'.format(_RECEIVER_FN_NAME, name)