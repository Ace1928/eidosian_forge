import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def _add_dispatch_for_binary_elementwise_api(api, x_type, y_type, elementwise_api_handler):
    """Registers a binary elementwise handler as a dispatcher for a given API."""
    api_signature = tf_inspect.signature(api)
    x_name, y_name = list(api_signature.parameters)[:2]
    name_index = _find_name_index(api_signature)
    need_to_bind_api_args = len(api_signature.parameters) > 3 or 'name' not in api_signature.parameters

    @dispatch_for_api(api, {x_name: x_type, y_name: y_type})
    def dispatch_target(*args, **kwargs):
        args, kwargs, name = _extract_name_arg(args, kwargs, name_index)
        if len(args) > 1:
            x, y, args = (args[0], args[1], args[2:])
        elif args:
            x, args = (args[0], args[1:])
            y = kwargs.pop(y_name, None)
        else:
            x = kwargs.pop(x_name, None)
            y = kwargs.pop(y_name, None)
        if need_to_bind_api_args:
            tensor_api = lambda v1, v2: api(v1, v2, *args, **kwargs)
        else:
            tensor_api = api
        if name is None:
            return elementwise_api_handler(tensor_api, x, y)
        else:
            with ops.name_scope(name, None, [x, y]):
                return elementwise_api_handler(tensor_api, x, y)
    dispatch_target.__name__ = 'elementwise_dispatch_target_for_' + api.__name__
    dispatch_target.__qualname__ = dispatch_target.__name__
    target_list = _ELEMENTWISE_API_TARGETS.setdefault((x_type, y_type), [])
    target_list.append((api, dispatch_target))