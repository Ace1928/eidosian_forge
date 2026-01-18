import functools
import threading
import weakref
import tensorflow.compat.v1.logging as logging
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.engine import input_spec
from keras.src.mixed_precision import autocast_variable
from keras.src.saving.legacy import saving_utils
from keras.src.saving.legacy.saved_model import constants
from keras.src.saving.legacy.saved_model import load as keras_load
from keras.src.saving.legacy.saved_model import serialized_attributes
from keras.src.saving.legacy.saved_model import utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_contextlib
from keras.src.utils import tf_utils
from keras.src.utils import version_utils
from keras.src.utils.generic_utils import LazyLoader
def _get_layer_inputs(self, layer):
    """Inspects layer object and returns the inferred input signature.

        Args:
          layer: Layer object.

        Returns:
          List of possibly nested TensorSpecs of the layer call function inputs
          in the form of `(args, kwargs)`
        """
    if isinstance(layer.call, tf.__internal__.function.Function) and layer.call.input_signature is not None:
        return (layer.call.input_signature, {})
    elif isinstance(layer, training_lib.Model):
        return saving_utils.model_call_inputs(layer)
    elif layer.input_spec is not None and layer._use_input_spec_as_call_signature:

        def to_tensor_spec_or_none(x):
            spec = input_spec.to_tensor_spec(x, layer._compute_dtype)
            if spec.shape == tf.TensorShape(None):
                return (None, None)
            return spec
        input_signature = [tf.nest.map_structure(to_tensor_spec_or_none, layer.input_spec)]
        return (input_signature, {})
    else:
        return (None, None)