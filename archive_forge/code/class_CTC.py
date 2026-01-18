import warnings
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import Loss
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.saving import serialization_lib
from keras.src.utils.numerical_utils import normalize
@keras_export('keras.losses.CTC')
class CTC(LossFunctionWrapper):
    """CTC (Connectionist Temporal Classification) loss.

    Args:
        y_true: A tensor of shape `(batch_size, target_max_length)` containing
            the true labels in integer format. `0` always represents
            the blank/mask index and should not be used for classes.
        y_pred: A tensor of shape `(batch_size, output_max_length, num_classes)`
            containing logits (the output of your model).
            They should *not* be normalized via softmax.
    """

    def __init__(self, reduction='sum_over_batch_size', name='sparse_categorical_crossentropy'):
        super().__init__(ctc, name=name, reduction=reduction)

    def get_config(self):
        return {'name': self.name, 'reduction': self.reduction}