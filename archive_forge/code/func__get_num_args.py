import itertools
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import resource_variable_ops
def _get_num_args(arg_def, node_def):
    if arg_def.number_attr:
        return node_def.attr[arg_def.number_attr].i
    elif arg_def.type_list_attr:
        return len(node_def.attr[arg_def.type_list_attr].list.type)
    elif arg_def.type_attr or arg_def.type != types_pb2.DT_INVALID:
        return 1
    else:
        raise ValueError(f'Invalid arg_def:\n\n{arg_def}. Please make sure the FunctionDef `fdef` is correct.')