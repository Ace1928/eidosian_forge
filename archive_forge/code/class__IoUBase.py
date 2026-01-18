from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils as dtensor_utils
from keras.src.metrics import base_metric
from tensorflow.python.util.tf_export import keras_export
class _IoUBase(base_metric.Metric):
    """Computes the confusion matrix for Intersection-Over-Union metrics.

    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    For an individual class, the IoU metric is defined as follows:

    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```

    From IoUs of individual classes, the MeanIoU can be computed as the mean of
    the individual IoUs.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      num_classes: The possible number of labels the prediction task can have.
        This value must be provided, since a confusion matrix of size
        `(num_classes, num_classes)` will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_true: Whether labels are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      sparse_y_pred: Whether predictions are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) -1 is the dimension containing the logits.
        Defaults to `-1`.
    """

    def __init__(self, num_classes: int, name: Optional[str]=None, dtype: Optional[Union[str, tf.dtypes.DType]]=None, ignore_class: Optional[int]=None, sparse_y_true: bool=True, sparse_y_pred: bool=True, axis: int=-1):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis
        self.total_cm = self.add_weight('total_confusion_matrix', shape=(num_classes, num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`. Defaults to `1`.

        Returns:
          Update op.
        """
        if not self.sparse_y_true:
            y_true = tf.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = tf.argmax(y_pred, axis=self.axis)
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])
        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])
        if self.ignore_class is not None:
            ignore_class = tf.cast(self.ignore_class, y_true.dtype)
            valid_mask = tf.not_equal(y_true, ignore_class)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[valid_mask]
        current_cm = tf.math.confusion_matrix(y_true, y_pred, self.num_classes, weights=sample_weight, dtype=self._dtype)
        return self.total_cm.assign_add(current_cm)

    def reset_state(self):
        backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))