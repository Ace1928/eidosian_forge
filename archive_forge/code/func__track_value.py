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
def _track_value(self, value, name):
    """Allows storage of non-trackable objects."""
    try:
        value = super()._track_value(value=value, name=name)
    except ValueError:
        value = sticky_attribute_assignment(trackable=self, value=value, name=name)
    return value