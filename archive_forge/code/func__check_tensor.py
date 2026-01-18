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
def _check_tensor(tensor, name, error_label='feature'):
    """Check that passed `tensor` is a Tensor or SparseTensor or RaggedTensor."""
    if not (isinstance(tensor, tf.Tensor) or isinstance(tensor, tf.sparse.SparseTensor) or isinstance(tensor, tf.RaggedTensor)):
        fmt_name = ' {}'.format(name) if name else ''
        value_error = ValueError('{}{} must be a Tensor, SparseTensor, or RaggedTensor.'.format(error_label, fmt_name))
        if hasattr(tensor, 'tensor'):
            try:
                ops.convert_to_tensor(tensor)
            except TypeError:
                raise value_error
        else:
            raise value_error