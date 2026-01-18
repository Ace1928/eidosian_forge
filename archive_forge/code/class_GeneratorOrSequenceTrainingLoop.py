import functools
import math
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class GeneratorOrSequenceTrainingLoop(training_utils_v1.TrainingLoop):
    """Generator-like.

  Input is Python generator, or Sequence object.

  The difference between this class and `GeneratorLikeTrainingFunction` is that
  this class only handles inputs that with x, y and sample_weight fused into one
  param.
  """

    def fit(self, model, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        model._validate_or_infer_batch_size(batch_size, steps_per_epoch, x)
        training_utils_v1.check_generator_arguments(y, sample_weight, validation_split=validation_split)
        return fit_generator(model, x, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=validation_data, validation_steps=validation_steps, validation_freq=validation_freq, class_weight=class_weight, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, shuffle=shuffle, initial_epoch=initial_epoch, steps_name='steps_per_epoch')

    def evaluate(self, model, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        model._validate_or_infer_batch_size(batch_size, steps, x)
        training_utils_v1.check_generator_arguments(y, sample_weight)
        return evaluate_generator(model, x, steps=steps, verbose=verbose, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

    def predict(self, model, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        model._validate_or_infer_batch_size(batch_size, steps, x)
        return predict_generator(model, x, steps=steps, verbose=verbose, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)