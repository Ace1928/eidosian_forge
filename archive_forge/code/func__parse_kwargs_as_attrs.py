import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
def _parse_kwargs_as_attrs(func_name, **kwargs):
    """Parses **kwargs into a node's attributes."""
    attrs = {}
    noinline = kwargs.pop('noinline', None)
    if noinline is not None:
        attrs['_noinline'] = attr_value_pb2.AttrValue(b=bool(noinline))
    attrs['_disable_call_shape_inference'] = attr_value_pb2.AttrValue(b=True)
    compiled = kwargs.pop('compiled', None)
    separate_compiled_gradients = kwargs.pop('separate_compiled_gradients', None)
    if compiled is not None:
        attrs['_XlaCompile'] = attr_value_pb2.AttrValue(b=bool(compiled))
        attrs['_XlaSeparateCompiledGradients'] = attr_value_pb2.AttrValue(b=bool(separate_compiled_gradients))
        if '_XlaScope' in ops.get_default_graph()._attr_scope_map:
            attrs['_XlaScope'] = ops.get_default_graph()._attr_scope_map['_XlaScope']
        else:
            attrs['_XlaScope'] = attr_value_pb2.AttrValue(s=('function_%s' % func_name).encode())
    kwargs_keys = list(kwargs.keys())
    for key in kwargs_keys:
        if key.startswith('experimental_'):
            attrs[key] = _get_experimental_kwarg_as_attr(key, kwargs[key])
            del kwargs[key]
        elif key == '_implements' or key == '_reference':
            attrs[key] = _get_kwarg_as_str_attr(key, kwargs[key])
            del kwargs[key]
    if kwargs:
        raise ValueError(f'Unknown keyword arguments: {kwargs.keys()}.')
    return attrs