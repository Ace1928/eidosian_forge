from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def _serialize_function_spec(function_spec):
    """Serialize a FunctionSpec object into its proto representation."""
    if function_spec.fullargspec.args and function_spec.fullargspec.args[0] == 'self':
        raise TypeError("Can not serialize tf.function with unbound 'self' parameter.")
    proto = saved_object_graph_pb2.FunctionSpec()
    proto.fullargspec.CopyFrom(nested_structure_coder.encode_structure(function_spec.fullargspec._replace(annotations={})))
    proto.is_method = False
    proto.input_signature.CopyFrom(nested_structure_coder.encode_structure(function_spec.input_signature))
    proto.jit_compile = {None: saved_object_graph_pb2.FunctionSpec.JitCompile.DEFAULT, True: saved_object_graph_pb2.FunctionSpec.JitCompile.ON, False: saved_object_graph_pb2.FunctionSpec.JitCompile.OFF}.get(function_spec.jit_compile)
    return proto