import collections
import contextlib
import functools
import itertools
import textwrap
import threading
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import lazy_variable
from keras.src.engine import base_layer_utils
from keras.src.engine import input_spec
from keras.src.engine import keras_tensor
from keras.src.engine import node as node_module
from keras.src.mixed_precision import autocast_variable
from keras.src.mixed_precision import policy
from keras.src.saving import serialization_lib
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import generic_utils
from keras.src.utils import layer_utils
from keras.src.utils import object_identity
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from keras.src.utils import version_utils
from keras.src.utils.generic_utils import to_snake_case  # noqa: F401
from keras.src.utils.tf_utils import is_tensor_or_tensor_list  # noqa: F401
from google.protobuf import json_format
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import (
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
@contextlib.contextmanager
def _name_scope_unnester(full_name_scope):
    """Helper to get relative name scope from fully-speced nested name scopes.

    Args:
      full_name_scope: full(absolute) name scope path.

    Yields:
      Relative name scope path from the parent `_name_scope_unnester` context
      manager.

    Example:
    ```
    with _name_scope_unnester('a') as name1:  # name1 == 'a'
      with _name_scope_unnester('a/b') as name2:  # name2 == 'b'
        with _name_scope_unnester('a/b/c') as name3:  # name3 == 'c'
          pass
    ```
    """
    if not getattr(_name_scope_unnester_stack, 'value', None):
        _name_scope_unnester_stack.value = ['']
    _name_scope_unnester_stack.value.append(full_name_scope)
    try:
        full_name_scope = _name_scope_unnester_stack.value[-1]
        outer_name_scope = _name_scope_unnester_stack.value[-2]
        relative_name_scope = full_name_scope.lstrip(outer_name_scope)
        relative_name_scope = relative_name_scope.lstrip('/')
        yield relative_name_scope
    finally:
        _name_scope_unnester_stack.value.pop()