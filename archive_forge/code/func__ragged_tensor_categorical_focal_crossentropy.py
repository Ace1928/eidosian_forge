import abc
import functools
import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.saving import saving_lib
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.utils import losses_utils
from keras.src.utils import tf_utils
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
@dispatch.dispatch_for_types(categorical_focal_crossentropy, tf.RaggedTensor)
def _ragged_tensor_categorical_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1):
    """Implements support for handling RaggedTensors.

    Expected shape: (batch, sequence_len, n_classes) with sequence_len
    being variable per batch.
    Return shape: (batch, sequence_len).
    When used by CategoricalFocalCrossentropy() with the default reduction
    (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
    number of elements independent of the batch. E.g. if the RaggedTensor
    has 2 batches with [2, 1] values respectively the resulting loss is
    the sum of the individual loss values divided by 3.

    Args:
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to -1. The dimension along which the entropy is
            computed.

    Returns:
      Categorical focal crossentropy loss value.
    """
    fn = functools.partial(categorical_focal_crossentropy, alpha=alpha, gamma=gamma, from_logits=from_logits, label_smoothing=label_smoothing, axis=axis)
    return _ragged_tensor_apply_loss(fn, y_true, y_pred)