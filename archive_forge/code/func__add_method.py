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
def _add_method(self, method_name, output_specs, input_specs, client_handle, doc_string):
    """Method to add RPC methods to the client object."""

    def validate_and_get_flat_inputs(*args):
        if args is None:
            args = []
        if input_specs:
            nest.assert_same_structure(args, input_specs)
        flat_inputs = nest.flatten(args)
        return flat_inputs

    def call_wrapper(*args, timeout_in_ms=0):
        status_or, deleter = gen_rpc_ops.rpc_call(client_handle, args=validate_and_get_flat_inputs(*args), method_name=method_name, timeout_in_ms=timeout_in_ms)
        return StatusOrResult(status_or, deleter, output_specs)

    def call_blocking_wrapper(*args, timeout_in_ms=0):
        status_or, deleter = gen_rpc_ops.rpc_call(client_handle, args=validate_and_get_flat_inputs(*args), method_name=method_name, timeout_in_ms=timeout_in_ms)
        status_or = StatusOrResult(status_or, deleter, output_specs)
        if status_or.is_ok():
            return status_or.get_value()
        else:
            error_code, error_msg = status_or.get_error()
            raise errors.exception_type_from_error_code(error_code.numpy())(None, None, error_msg.numpy())
    setattr(self, method_name, call_wrapper)
    call_wrapper.__doc__ = doc_string
    blocking_method_name = method_name + '_blocking'
    setattr(self, blocking_method_name, call_blocking_wrapper)
    call_blocking_wrapper.__doc__ = doc_string