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
def _dispatch_to_method_handler(self, attr_name, self_arg, is_unbound_method, type_name, node, function, arg_list, kwargs):
    method_handler = self._find_handler('method_%s_%s' % (type_name, attr_name), kwargs)
    if method_handler is None:
        if attr_name in TypeSlots.special_method_names or attr_name in ['__new__', '__class__']:
            method_handler = self._find_handler('slot%s' % attr_name, kwargs)
        if method_handler is None:
            return self._handle_method(node, type_name, attr_name, function, arg_list, is_unbound_method, kwargs)
    if self_arg is not None:
        arg_list = [self_arg] + list(arg_list)
    if kwargs:
        result = method_handler(node, function, arg_list, is_unbound_method, kwargs)
    else:
        result = method_handler(node, function, arg_list, is_unbound_method)
    return result