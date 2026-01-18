import copy
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn import rnn_utils
from keras.src.layers.rnn.base_wrapper import Wrapper
from keras.src.saving import serialization_lib
from keras.src.utils import generic_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
def _verify_layer_config(self):
    """Ensure the forward and backward layers have valid common property."""
    if self.forward_layer.go_backwards == self.backward_layer.go_backwards:
        raise ValueError(f'Forward layer and backward layer should have different `go_backwards` value.forward_layer.go_backwards = {self.forward_layer.go_backwards},backward_layer.go_backwards = {self.backward_layer.go_backwards}')
    common_attributes = ('stateful', 'return_sequences', 'return_state')
    for a in common_attributes:
        forward_value = getattr(self.forward_layer, a)
        backward_value = getattr(self.backward_layer, a)
        if forward_value != backward_value:
            raise ValueError(f'Forward layer and backward layer are expected to have the same value for attribute "{a}", got "{forward_value}" for forward layer and "{backward_value}" for backward layer')