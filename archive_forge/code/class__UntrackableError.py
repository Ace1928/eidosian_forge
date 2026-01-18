import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _UntrackableError(ValueError):

    def __init__(self, value):
        self._value = value

    def __str__(self):
        return f'Only trackable objects (such as Layers or Optimizers) may be stored in a List object. Got {self._value}, which does not inherit from Trackable.'