import collections
import copy
import itertools
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class ModuleWrapper(base_layer.Layer):
    """Wrapper for `tf.Module`s to support the Functional and Sequential API."""

    def __init__(self, module, method_name=None, **kwargs):
        """Initializes the wrapper Layer for this module.

    Args:
      module: The `tf.Module` instance to be wrapped.
      method_name: (Optional) str. The name of the method to use as the forward
        pass of the module. If not set, defaults to '__call__' if defined, or
        'call'.
      **kwargs: Additional keywrod arguments. See `tf.keras.layers.Layer`.

    Raises:
      ValueError: If `method` is not defined on `module`.
    """
        super(ModuleWrapper, self).__init__(**kwargs)
        if method_name is None:
            if hasattr(module, '__call__'):
                method_name = '__call__'
            elif hasattr(module, 'call'):
                method_name = 'call'
        if method_name is None or not hasattr(module, method_name):
            raise ValueError('{} is not defined on object {}'.format(method_name, module))
        self._module = module
        self._method_name = method_name
        method = getattr(module, method_name)
        method_arg_spec = tf_inspect.getfullargspec(method)
        self._expects_training_arg = 'training' in method_arg_spec.args or method_arg_spec.varkw is not None
        self._expects_mask_arg = 'mask' in method_arg_spec.args or method_arg_spec.varkw is not None

    def call(self, *args, **kwargs):
        if 'training' in kwargs and (not self._expects_training_arg):
            kwargs.pop('training')
        if 'mask' in kwargs and (not self._expects_mask_arg):
            kwargs.pop('mask')
        return getattr(self._module, self._method_name)(*args, **kwargs)