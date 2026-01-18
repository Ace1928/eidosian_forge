import functools
import threading
import weakref
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def _generate_input_signature(self, layer):
    """Inspects layer object and returns the inferred input signature.

    Args:
      layer: Layer object.

    Returns:
      List of possibly nested TensorSpecs of the layer call function inputs.
      The list does not contain the `training` argument.
    """
    if isinstance(layer.call, def_function.Function) and layer.call.input_signature is not None:
        return layer.call.input_signature
    elif isinstance(layer, training_lib.Model):
        return saving_utils.model_input_signature(layer)
    elif layer.input_spec is not None and layer._use_input_spec_as_call_signature:

        def to_tensor_spec_or_none(x):
            spec = input_spec.to_tensor_spec(x, layer._compute_dtype)
            if spec.shape == tensor_shape.TensorShape(None):
                return None
            return spec
        input_signature = [nest.map_structure(to_tensor_spec_or_none, layer.input_spec)]
        return input_signature
    else:
        return None