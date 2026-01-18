from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def _validate_inputs(concrete_function):
    """Raises error if input type is tf.Variable."""
    if any((isinstance(inp, resource_variable_ops.VariableSpec) for inp in nest.flatten(concrete_function.structured_input_signature))):
        raise ValueError(f"Unable to serialize concrete_function '{concrete_function.name}'with tf.Variable input. Functions that expect tf.Variable inputs cannot be exported as signatures.")