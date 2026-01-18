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
class LayerCall(object):
    """Function that triggers traces of other functions in the same collection."""

    def __init__(self, call_collection, call_fn, name, input_signature):
        """Initializes a LayerCall object.

    Args:
      call_collection: a LayerCallCollection, which contains the other layer
        call functions (e.g. call_with_conditional_losses, call). These
        functions should be traced with the same arguments.
      call_fn: A call function.
      name: Name of the call function.
      input_signature: Input signature of call_fn (can be None).
    """
        self.call_collection = call_collection
        self.input_signature = input_signature
        self.wrapped_call = def_function.function(layer_call_wrapper(call_collection, call_fn, name), input_signature=input_signature)
        self.original_layer_call = call_collection.layer_call_method

    def _maybe_trace(self, args, kwargs):
        if tracing_enabled():
            self.call_collection.add_trace(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._maybe_trace(args, kwargs)
        return self.wrapped_call(*args, **kwargs)

    def get_concrete_function(self, *args, **kwargs):
        self._maybe_trace(args, kwargs)
        return self.wrapped_call.get_concrete_function(*args, **kwargs)