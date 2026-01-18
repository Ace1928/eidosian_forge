import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref
import numpy as np
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
class CustomObjectScope(object):
    """Exposes custom classes/functions to Keras deserialization internals.

  Under a scope `with custom_object_scope(objects_dict)`, Keras methods such
  as `tf.keras.models.load_model` or `tf.keras.models.model_from_config`
  will be able to deserialize any custom object referenced by a
  saved config (e.g. a custom layer or metric).

  Example:

  Consider a custom regularizer `my_regularizer`:

  ```python
  layer = Dense(3, kernel_regularizer=my_regularizer)
  config = layer.get_config()  # Config contains a reference to `my_regularizer`
  ...
  # Later:
  with custom_object_scope({'my_regularizer': my_regularizer}):
    layer = Dense.from_config(config)
  ```

  Args:
      *args: Dictionary or dictionaries of `{name: object}` pairs.
  """

    def __init__(self, *args):
        self.custom_objects = args
        self.backup = None

    def __enter__(self):
        self.backup = _GLOBAL_CUSTOM_OBJECTS.copy()
        for objects in self.custom_objects:
            _GLOBAL_CUSTOM_OBJECTS.update(objects)
        return self

    def __exit__(self, *args, **kwargs):
        _GLOBAL_CUSTOM_OBJECTS.clear()
        _GLOBAL_CUSTOM_OBJECTS.update(self.backup)