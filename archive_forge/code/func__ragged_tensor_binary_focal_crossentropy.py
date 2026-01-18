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
@dispatch.dispatch_for_types(binary_focal_crossentropy, tf.RaggedTensor)
def _ragged_tensor_binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=False, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1):
    """Implements support for handling RaggedTensors.

    Expected shape: `(batch, sequence_len)` with sequence_len being variable per
    batch.
    Return shape: `(batch,)`; returns the per batch mean of the loss values.

    When used by BinaryFocalCrossentropy() with the default reduction
    (SUM_OVER_BATCH_SIZE), the reduction averages the per batch losses over
    the number of batches.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        apply_class_balancing: A bool, whether to apply weight balancing on the
            binary classes 0 and 1.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in the reference [Lin et al., 2018](
            https://arxiv.org/pdf/1708.02002.pdf). The weight for class 0 is
            `1.0 - alpha`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Axis along which to compute crossentropy.

    Returns:
        Binary focal crossentropy loss value.
    """
    fn = functools.partial(binary_focal_crossentropy, apply_class_balancing=apply_class_balancing, alpha=alpha, gamma=gamma, from_logits=from_logits, label_smoothing=label_smoothing, axis=axis)
    return _ragged_tensor_apply_loss(fn, y_true, y_pred)