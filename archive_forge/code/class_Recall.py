import abc
import types
import warnings
import numpy as np
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.saving.saved_model import metric_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class Recall(Metric):
    """Computes the recall of the predictions with respect to the labels.

  This metric creates two local variables, `true_positives` and
  `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives` and `false_negatives`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `top_k` is set, recall will be computed as how often on average a class
  among the labels of a batch entry is in the top-k predictions.

  If `class_id` is specified, we calculate recall by considering only the
  entries in the batch for which `class_id` is in the label, and computing the
  fraction of them for which `class_id` is above the threshold and/or in the
  top-k predictions.

  Args:
    thresholds: (Optional) A float value or a python list/tuple of float
      threshold values in [0, 1]. A threshold is compared with prediction
      values to determine the truth value of predictions (i.e., above the
      threshold is `true`, below is `false`). One metric value is generated
      for each threshold value. If neither thresholds nor top_k are set, the
      default is to calculate recall with `thresholds=0.5`.
    top_k: (Optional) Unset by default. An int value specifying the top-k
      predictions to consider when calculating recall.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Recall()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
  >>> m.result().numpy()
  0.6666667

  >>> m.reset_state()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.Recall()])
  ```
  """

    def __init__(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        super(Recall, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(thresholds, default_threshold=default_threshold)
        self._thresholds_distributed_evenly = metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        self.true_positives = self.add_weight('true_positives', shape=(len(self.thresholds),), initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight('false_negatives', shape=(len(self.thresholds),), initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.

    Args:
      y_true: The ground truth values, with the same dimensions as `y_pred`.
        Will be cast to `bool`.
      y_pred: The predicted values. Each element must be in the range `[0, 1]`.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
        return metrics_utils.update_confusion_matrix_variables({metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives, metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives}, y_true, y_pred, thresholds=self.thresholds, thresholds_distributed_evenly=self._thresholds_distributed_evenly, top_k=self.top_k, class_id=self.class_id, sample_weight=sample_weight)

    def result(self):
        result = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value([(v, np.zeros((num_thresholds,))) for v in (self.true_positives, self.false_negatives)])

    def get_config(self):
        config = {'thresholds': self.init_thresholds, 'top_k': self.top_k, 'class_id': self.class_id}
        base_config = super(Recall, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))