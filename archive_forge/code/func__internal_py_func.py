import functools
import threading
import traceback  # pylint: disable=unused-import
import weakref
import numpy as np
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.lib.core import _pywrap_py_func
from tensorflow.python.ops import autograph_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _internal_py_func(func, inp, Tout, stateful=None, use_eager_py_func=False, is_grad_func=False, name=None):
    """See documentation for py_func and eager_py_func."""
    if not callable(func):
        raise ValueError(f'Expected func to be callable. Received func={func} of type {type(func)}.')
    original_func = func
    func = autograph.do_not_convert(func)
    inp = variable_utils.convert_variables_to_tensors(list(inp))
    is_list_or_tuple = isinstance(Tout, (list, tuple))
    Tout = Tout if is_list_or_tuple else [Tout]
    Tout = [_as_dtype_or_type_spec(t) for t in Tout]
    handle_composite_tensors = use_eager_py_func and (any((isinstance(v, composite_tensor.CompositeTensor) for v in inp)) or any((isinstance(t, type_spec.TypeSpec) for t in Tout)))
    if handle_composite_tensors:
        func, inp, Tout, out_structure = _wrap_for_composites(func, inp, Tout)
    if use_eager_py_func:
        func = EagerFunc(func, Tout, is_grad_func)
    if tf_inspect.isfunction(original_func):
        original_func.ag_dnc_wrapper__ = func
    token = _py_funcs.insert(func)
    graph = ops.get_default_graph()
    while True:
        current_graph = graph
        if isinstance(graph, function._FuncGraph):
            graph = graph._outer_graph
        elif isinstance(graph, func_graph.FuncGraph):
            graph = graph.outer_graph
        if graph is current_graph:
            break
    if not hasattr(graph, '_py_funcs_used_in_graph'):
        graph._py_funcs_used_in_graph = []
    graph._py_funcs_used_in_graph.append(func)
    if use_eager_py_func:
        result = gen_script_ops.eager_py_func(input=inp, token=token, is_async=context.is_async(), Tout=Tout, name=name)
    elif stateful:
        result = gen_script_ops.py_func(input=inp, token=token, Tout=Tout, name=name)
    else:
        result = gen_script_ops.py_func_stateless(input=inp, token=token, Tout=Tout, name=name)
    if handle_composite_tensors and Tout:
        result = nest.pack_sequence_as(out_structure, result, expand_composites=True)
    return result if is_list_or_tuple else result[0]