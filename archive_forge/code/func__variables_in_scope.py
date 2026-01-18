import functools
import traceback
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _variables_in_scope(self, variable_list):
    if self._variable_scope_name is None:
        raise RuntimeError('A variable scope must be set before variables can be accessed.')
    return [v for v in variable_list if v.name.startswith(self._variable_scope_name + '/')]