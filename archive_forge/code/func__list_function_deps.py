import collections
import pprint
import re
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def _list_function_deps(fdef, library_function_names, library_gradient_names):
    """Find functions referenced in `fdef`."""
    deps = set()
    for node_def in fdef.node_def:
        grad_op_type = _get_gradient_op_type(node_def)
        if node_def.op in library_function_names:
            deps.add(node_def.op)
        elif grad_op_type and grad_op_type in library_gradient_names:
            deps.add(library_gradient_names[grad_op_type])
        else:
            for _, attr_value in node_def.attr.items():
                if attr_value.WhichOneof('value') == 'func':
                    deps.add(attr_value.func.name)
                elif attr_value.WhichOneof('value') == 'list':
                    for fn in attr_value.list.func:
                        deps.add(fn.name)
    return deps