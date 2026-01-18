from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _check_tensor_key(name, error_label='feature', allow_ints=False):
    if not isinstance(name, six.string_types):
        if not allow_ints:
            raise ValueError('{} keys must be strings: {}.'.format(error_label, name))
        elif not isinstance(name, six.integer_types):
            raise ValueError('{} keys must be strings or ints: {}.'.format(error_label, name))