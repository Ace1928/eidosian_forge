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
@property
def _dirty(self):
    """Check if there has already been a mutation which prevents saving."""
    return self._self_external_modification or self._self_non_string_key