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
class _VariableScopeStore(threading.local):
    """A thread local store for the current variable scope and scope counts."""

    def __init__(self):
        super(_VariableScopeStore, self).__init__()
        self.current_scope = VariableScope(False)
        self.variable_scopes_count = {}

    def open_variable_scope(self, scope_name):
        if scope_name in self.variable_scopes_count:
            self.variable_scopes_count[scope_name] += 1
        else:
            self.variable_scopes_count[scope_name] = 1

    def close_variable_subscopes(self, scope_name):
        if scope_name is None:
            for k in self.variable_scopes_count:
                self.variable_scopes_count[k] = 0
        else:
            startswith_check = scope_name + '/'
            startswith_len = len(startswith_check)
            for k in self.variable_scopes_count:
                if k[:startswith_len] == startswith_check:
                    self.variable_scopes_count[k] = 0

    def variable_scope_count(self, scope_name):
        return self.variable_scopes_count.get(scope_name, 0)