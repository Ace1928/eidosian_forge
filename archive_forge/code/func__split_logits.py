from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _split_logits(self, logits):
    """Splits logits along the last dimension and returns a dict.

    If the input logits is not a dict, splitting is applied based on the logits
    dimension of each head.
    For example:

    ```python
    # head1.logits_dimension = 2
    # head2.logits_dimension = 3
    head1 = tf.estimator.MultiLabelHead(n_classes=2, name='head1_name')
    head2 = tf.estimator.MultiClassHead(n_classes=3, name='head2_name')
    multi_head = tf.estimator.MultiHead([head1, head2])
    # Input logits
    logits = np.array([[-1., 1., 2., -2., 2.], [-1.5, 1., -3., 2., -2.]],
                      dtype=np.float32)
    # As logits is not a dict, _split_logits is applied and returns the
    # logits_dict as
    logits_dict = {'head1_name': [[-1., 1.], [-1.5, 1.]],
                   'head2_name':  [[2., -2., 2.], [-3., 2., -2.]]}
    ```
    Args:
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.

    Returns:
      logits_dict: A dict of logits for each head.
    """
    logits_dict = {}
    with ops.name_scope('split_logits', values=[logits]):
        logits = ops.convert_to_tensor(logits)
        logits_dimensions = [head.logits_dimension for head in self._heads]
        total_logits_dimension = sum(logits_dimensions)
        logits_tensor_shape = logits.shape.as_list()
        last_dimension_size = logits_tensor_shape[-1]
        if last_dimension_size is not None:
            if last_dimension_size != total_logits_dimension:
                raise ValueError('Could not split logits of shape %r among the heads with individual logits dimensions: %r. The last dimension of the logits tensor should equal %d but is %d.' % (logits_tensor_shape, logits_dimensions, last_dimension_size, total_logits_dimension))
        if tf.executing_eagerly():
            logits_shape = logits._shape_tuple()
            batch_shape = logits_shape[:-1]
        else:
            batch_shape = tf.compat.v1.shape(logits)[:-1]
        zeros_like_batch_shape = tf.compat.v1.zeros_like(batch_shape)
        minus_ones_like_batch_shape = -1 * tf.compat.v1.ones_like(batch_shape)
        begin_idx = 0
        for head in self._heads:
            begin_tensor = tf.concat([zeros_like_batch_shape, [begin_idx]], axis=0)
            size_tensor = tf.concat([minus_ones_like_batch_shape, [head.logits_dimension]], axis=0)
            logits_dict[head.name] = tf.slice(logits, begin=begin_tensor, size=size_tensor)
            begin_idx += head.logits_dimension
    return logits_dict