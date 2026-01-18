from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
def _delegate_to_assigned_value(self, node, function, arg_list, kwargs):
    assignment = function.cf_state[0]
    value = assignment.rhs
    if value.is_name:
        if not value.entry or len(value.entry.cf_assignments) > 1:
            return node
    elif value.is_attribute and value.obj.is_name:
        if not value.obj.entry or len(value.obj.entry.cf_assignments) > 1:
            return node
    else:
        return node
    return self._dispatch_to_handler(node, value, arg_list, kwargs)