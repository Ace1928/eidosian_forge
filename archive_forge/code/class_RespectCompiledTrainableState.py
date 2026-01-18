import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
class RespectCompiledTrainableState(object):
    """Set and restore trainable state if it has changed since compile.

  The keras API guarantees that the value of each Layer's `trainable` property
  at `Model.compile` time will be used when training that model. In order to
  respect this requirement, it may be necessary to set the trainable value of
  layers to their compile time values before beginning a training endpoint and
  restore the values before returing from said endpoint. This scope checks if
  any layer's trainable state has changed since Model compile, and performs this
  set and un-set bookkeeping.

  However, the trainable state of a layer changes quite infrequently, if ever,
  for many kinds of workflows. Moreover, updating every layer in a model is an
  expensive operation. As a result, we will only explicitly set and unset the
  trainable state of a model if a trainable value has changed since compile.
  """

    def __init__(self, model):
        self._model = model
        self._current_trainable_state = None
        self._compiled_trainable_state = None
        self._should_set_trainable = False

    def __enter__(self):
        self._current_trainable_state = self._model._get_trainable_state()
        self._compiled_trainable_state = self._model._compiled_trainable_state
        for layer, trainable in self._compiled_trainable_state.items():
            if layer in self._current_trainable_state and trainable != self._current_trainable_state[layer]:
                self._should_set_trainable = True
                break
        if self._should_set_trainable:
            self._model._set_trainable_state(self._compiled_trainable_state)

    def __exit__(self, type_arg, value_arg, traceback_arg):
        if self._should_set_trainable:
            self._model._set_trainable_state(self._current_trainable_state)
        return False