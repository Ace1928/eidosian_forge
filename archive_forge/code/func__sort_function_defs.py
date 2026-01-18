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
def _sort_function_defs(library, function_deps):
    """Return a topologic sort of FunctionDefs in a library."""
    edges = collections.defaultdict(list)
    in_count = collections.defaultdict(lambda: 0)
    for fname, deps in function_deps.items():
        for dep in deps:
            edges[dep].append(fname)
            in_count[fname] += 1
    ready = [fdef.signature.name for fdef in library.function if in_count[fdef.signature.name] == 0]
    output = []
    while ready:
        node = ready.pop()
        output.append(node)
        for dest in edges[node]:
            in_count[dest] -= 1
            if not in_count[dest]:
                ready.append(dest)
    if len(output) != len(library.function):
        failed_to_resolve = sorted(set(in_count.keys()) - set(output))
        raise ValueError('There is a cyclic dependency between functions. ', f'Could not resolve {failed_to_resolve}.')
    reverse = {fdef.signature.name: fdef for fdef in library.function}
    return [reverse[x] for x in output]