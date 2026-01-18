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
def add_function(self, call_fn, name, match_layer_training_arg):
    """Adds a layer call function to the collection.

    Args:
      call_fn: a python function
      name: Name of call function
      match_layer_training_arg: If True, removes the `training` from the
        function arguments when calling `call_fn`.

    Returns:
      LayerCall (tf.function)
    """
    fn = LayerCall(self, self._maybe_wrap_with_training_arg(call_fn, match_layer_training_arg), name, input_signature=self.fn_input_signature)
    self._functions[name] = fn.wrapped_call
    return fn