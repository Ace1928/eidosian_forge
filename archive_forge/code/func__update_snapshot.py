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
def _update_snapshot(self):
    """Acknowledges tracked changes to the wrapped dict."""
    self._attribute_sentinel.invalidate_all()
    if self._dirty:
        return
    self._self_last_wrapped_dict_snapshot = dict(self)