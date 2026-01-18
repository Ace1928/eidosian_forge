from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import concrete_function
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import gradients_util
from tensorflow.python.util import keras_deps
from tensorflow.python.util import tf_contextlib
def _extract_input_index(function_attribute_name):
    func_name = node_def.attr[function_attribute_name].func.name
    fdef = functions[func_name].cached_definition
    output_arg_name = fdef.signature.output_arg[output_idx].name
    output_tensor_name = fdef.ret[output_arg_name]
    return resource_input_index(output_tensor_name, [arg.name for arg in fdef.signature.input_arg], {ndef.name: ndef for ndef in fdef.node_def}, functions)