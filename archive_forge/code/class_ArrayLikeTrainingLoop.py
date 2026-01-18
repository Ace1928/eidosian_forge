import functools
import numpy as np
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class ArrayLikeTrainingLoop(training_utils_v1.TrainingLoop):
    """TrainingLoop that handle inputs like array.

  This is the default handler for most of the input data types, includes
  symbolic tensors or Numpy array-like, Datasets and iterators in graph mode
  (since they generate symbolic tensors). This Function is used to handle model
  with `run_eagerly` = False.
  """

    def fit(self, model, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, **kwargs):
        batch_size = model._validate_or_infer_batch_size(batch_size, steps_per_epoch, x)
        x, y, sample_weights = model._standardize_user_data(x, y, sample_weight=sample_weight, class_weight=class_weight, batch_size=batch_size, check_steps=True, steps_name='steps_per_epoch', steps=steps_per_epoch, validation_split=validation_split, shuffle=shuffle)
        if validation_data:
            val_x, val_y, val_sample_weights = model._prepare_validation_data(validation_data, batch_size, validation_steps)
        elif validation_split and 0.0 < validation_split < 1.0:
            x, y, sample_weights, val_x, val_y, val_sample_weights = training_utils_v1.split_training_and_validation_data(x, y, sample_weights, validation_split)
        else:
            if validation_steps:
                raise ValueError('`validation_steps` should not be specified if `validation_data` is None.')
            val_x, val_y, val_sample_weights = (None, None, None)
        return fit_loop(model, inputs=x, targets=y, sample_weights=sample_weights, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, val_inputs=val_x, val_targets=val_y, val_sample_weights=val_sample_weights, shuffle=shuffle, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq, steps_name='steps_per_epoch')

    def evaluate(self, model, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, **kwargs):
        batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
        x, y, sample_weights = model._standardize_user_data(x, y, sample_weight=sample_weight, batch_size=batch_size, check_steps=True, steps_name='steps', steps=steps)
        return test_loop(model, inputs=x, targets=y, sample_weights=sample_weights, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)

    def predict(self, model, x, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
        x, _, _ = model._standardize_user_data(x, check_steps=True, steps_name='steps', steps=steps)
        return predict_loop(model, x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)