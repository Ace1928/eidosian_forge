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
def _valid_signature(concrete_function):
    """Returns whether concrete function can be converted to a signature."""
    if not concrete_function.outputs:
        return False
    try:
        _validate_inputs(concrete_function)
        _normalize_outputs(concrete_function.structured_outputs, 'unused', 'unused')
    except ValueError:
        return False
    return True