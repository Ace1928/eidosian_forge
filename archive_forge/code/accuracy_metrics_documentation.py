import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils as dtensor_utils
from keras.src.metrics import base_metric
from keras.src.utils import metrics_utils
from tensorflow.python.util.tf_export import keras_export
Computes how often integer targets are in the top `K` predictions.

    Standalone usage:
    >>> y_true = [2, 1]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(
    ...     y_true, y_pred, k=3)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([1., 1.], dtype=float32)

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to `5`.

    Returns:
      Sparse top K categorical accuracy value.
    