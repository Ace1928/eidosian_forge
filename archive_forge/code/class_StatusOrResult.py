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
class StatusOrResult(object):
    """Class representing result and status from RPC Call."""

    def __init__(self, status_or, deleter, output_specs=None):
        self._status_or = status_or
        self._output_specs = output_specs
        self._deleter = deleter
        self._error_code: dtypes.int64 = None
        self._error_message: dtypes.string = None

    def _check_status(self):
        if self._error_code is None:
            self._error_code, self._error_message = gen_rpc_ops.rpc_check_status(self._status_or)

    def __del__(self):
        if context.executing_eagerly():
            with context.eager_mode():
                gen_rpc_ops.delete_rpc_future_resource(handle=self._status_or, deleter=self._deleter)
        else:
            with context.graph_mode():
                gen_rpc_ops.delete_rpc_future_resource(handle=self._status_or, deleter=self._deleter)

    def is_ok(self):
        """Returns True if RPC is successful, otherwise returns False.

    This call will block for RPC result.
    """
        self._check_status()
        return math_ops.equal(self._error_code, constant_op.constant(0, dtype=dtypes.int64))

    def get_error(self):
        """Returns (TF Error Code, Error Message) from RPC Response.

    This call will block for RPC result.
    """
        self._check_status()
        return (self._error_code, self._error_message)

    def get_value(self):
        """Returns the returned response value from RPC Call when RPC is successful.

      The returned value is tensors in the output_specs format as returned from
      the RPC call


    This call will block for RPC result.
    """
        self._check_status()
        if self._output_specs is None or isinstance(self._output_specs, structure.NoneTensorSpec):
            flat_output_dtypes = []
            return_none = True
        else:
            return_none = False
            flat_output_dtypes = [s.dtype for s in nest.flatten(self._output_specs)]
        result = gen_rpc_ops.rpc_get_value(self._status_or, Tout=flat_output_dtypes)
        if return_none:
            return None
        else:
            return nest.pack_sequence_as(self._output_specs, result)