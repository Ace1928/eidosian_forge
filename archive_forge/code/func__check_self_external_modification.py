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
def _check_self_external_modification(self):
    """Checks for any changes to the wrapped dict not through the wrapper."""
    if self._dirty:
        return
    if self != self._self_last_wrapped_dict_snapshot:
        self._self_external_modification = True
        self._self_last_wrapped_dict_snapshot = None