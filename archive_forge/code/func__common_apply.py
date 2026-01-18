from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _common_apply(self, grads, trainable_variables=None):
    finite = self.check_finite(grads)
    ops.cond(finite, lambda: self._stateful_handle_finite_grads(grads, trainable_variables), self._stateful_handle_non_finite_grads)