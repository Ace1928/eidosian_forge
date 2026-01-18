from typing import Optional, Sequence, Union
import tensorflow.distribute.experimental.rpc.kernels.gen_rpc_ops as gen_rpc_ops
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as rpc_pb2
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class GrpcServer(Server):
    """GrpcServer object encapsulates a resource with GRPC server.

    Functions can be registered locally and are exposed via RPCs.
    Example:
    ```
    server = rpc_ops.GrpcServer("host:port")
    @tf.function
    def add(a, b):
      return a + b

    server.register("add", add)
    server.start()
    ```
  """

    def __init__(self, address: str):
        self._server_handle = gen_rpc_ops.rpc_server(address)
        if context.executing_eagerly():
            self._handle_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._server_handle, handle_device=self._server_handle.device)
        else:
            raise NotImplementedError('Please create the server outside tf.function.')

    def register(self, method_name: str, func: Union[def_function.Function, tf_function.ConcreteFunction]):
        """Method for registering functions."""
        if isinstance(func, def_function.Function):
            if func.function_spec.arg_names:
                if func.input_signature is None:
                    raise ValueError('Input signature not specified for the function.')
            concrete_fn = func.get_concrete_function()
            gen_rpc_ops.rpc_server_register(self._server_handle, method_name=method_name, captured_inputs=concrete_fn.captured_inputs, input_specs=get_input_specs_from_function(concrete_fn), output_specs=get_output_specs_from_function(concrete_fn), f=concrete_fn)
        elif isinstance(func, tf_function.ConcreteFunction):
            gen_rpc_ops.rpc_server_register(self._server_handle, method_name=method_name, captured_inputs=func.captured_inputs, input_specs=get_input_specs_from_function(func), output_specs=get_output_specs_from_function(func), f=func)
        else:
            raise ValueError('Only TF functions are supported with Register method')

    def start(self):
        """Starts GRPC server."""
        gen_rpc_ops.rpc_server_start(self._server_handle)