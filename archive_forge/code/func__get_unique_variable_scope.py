import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _get_unique_variable_scope(prefix):
    """Get a name with the given prefix unique in the current variable scope."""
    var_scope_store = get_variable_scope_store()
    current_scope = get_variable_scope()
    name = current_scope.name + '/' + prefix if current_scope.name else prefix
    if var_scope_store.variable_scope_count(name) == 0:
        return prefix
    idx = 1
    while var_scope_store.variable_scope_count(name + '_%d' % idx) > 0:
        idx += 1
    return prefix + '_%d' % idx