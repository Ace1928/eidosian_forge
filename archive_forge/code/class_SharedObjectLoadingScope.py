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
class SharedObjectLoadingScope(object):
    """A context manager for keeping track of loaded objects.

  During the deserialization process, we may come across objects that are
  shared across multiple layers. In order to accurately restore the network
  structure to its original state, `SharedObjectLoadingScope` allows us to
  re-use shared objects rather than cloning them.
  """

    def __enter__(self):
        if _shared_object_disabled():
            return NoopLoadingScope()
        global SHARED_OBJECT_LOADING
        SHARED_OBJECT_LOADING.scope = self
        self._obj_ids_to_obj = {}
        return self

    def get(self, object_id):
        """Given a shared object ID, returns a previously instantiated object.

    Args:
      object_id: shared object ID to use when attempting to find already-loaded
        object.

    Returns:
      The object, if we've seen this ID before. Else, `None`.
    """
        if object_id is None:
            return
        return self._obj_ids_to_obj.get(object_id)

    def set(self, object_id, obj):
        """Stores an instantiated object for future lookup and sharing."""
        if object_id is None:
            return
        self._obj_ids_to_obj[object_id] = obj

    def __exit__(self, *args, **kwargs):
        global SHARED_OBJECT_LOADING
        SHARED_OBJECT_LOADING.scope = NoopLoadingScope()