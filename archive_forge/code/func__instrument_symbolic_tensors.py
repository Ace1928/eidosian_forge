import atexit
import os
import re
import socket
import threading
import uuid
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _instrument_symbolic_tensors(self, tensors, op_type, op_name, tfdbg_context_id, tensor_ids):
    """Add debugging instrumentation for symbolic (i.e., non-eager) tensors.

    The detailed fashion in which the tensors are instrumented is determined
    by the tensor_debug_mode configured for the currently enabled dumping
    callback.

    Args:
      tensors: A tuple of Tensors to instrument. It is assumed that their
        ordering corresponds to the ordering of output tensors of an original
        op. Output slot indices (0-based) will be generated based on the
        ordering.
      op_type: Type name of the op that emits the Tensors (e.g., "MatMul").
      op_name: Name of the op that emits the Tensors (e.g., "dense_1/MatMul").
      tfdbg_context_id: A unique ID for the context that the op belongs to
        (e.g., a graph).
      tensor_ids: A list of unique ID numbers for the tensors, for tfdbg's
        internal use.

    Returns:
      Non-eager Tensors that override the `tensors` as the output of the op
      that originally generated `tensors`. In some cases (e.g., non-V1 graph
      mode), this may be `None`, as the instrumentation can simply rely on
      automatic control dependencies (see `auto_control_deps.py`) instead of
      tensor overriding.
    """
    tensor_debug_mode = self._tensor_debug_mode
    debug_urls = ['file://%s' % self._dump_root]
    is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
    instrumented_tensors = [] if is_v1_graph_mode else None
    for output_slot, tensor in enumerate(tensors):
        with self._symbolic_tensor_counter_lock:
            debug_identity_name = 'DebugIdentityV2_%d' % self._symbolic_tensor_counter
        debug_identity_op_kwargs = {'tfdbg_context_id': tfdbg_context_id, 'op_name': op_name, 'output_slot': output_slot, 'tensor_debug_mode': self._tensor_debug_mode, 'debug_urls': debug_urls, 'name': debug_identity_name, 'circular_buffer_size': self._circular_buffer_size, 'tfdbg_run_id': self._tfdbg_run_id}
        if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
            if not self._should_dump_tensor(op_type, tensor.dtype) or not tensor.dtype.is_numpy_compatible:
                if is_v1_graph_mode:
                    instrumented_tensors.append(tensor)
                continue
            if is_v1_graph_mode and (not tensor.dtype.is_numpy_compatible):
                instrumented_tensors.append(tensor)
                continue
            debug_tensor = gen_debug_ops.debug_identity_v2(constant_op.constant([], dtype=dtypes.float32), **debug_identity_op_kwargs)
            if is_v1_graph_mode:
                instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
        elif tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH, debug_event_pb2.TensorDebugMode.SHAPE):
            dtype = tensor.dtype
            dtype_is_dumpable = tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH) and dtype.is_floating or (tensor_debug_mode == debug_event_pb2.TensorDebugMode.SHAPE and (dtype.is_floating or dtype.is_integer or dtype.is_bool))
            if not self._should_dump_tensor(op_type, tensor.dtype) or not dtype_is_dumpable:
                if is_v1_graph_mode:
                    instrumented_tensors.append(tensor)
                continue
            debug_tensor = gen_debug_ops.debug_identity_v2(gen_debug_ops.debug_numeric_summary_v2(tensor, tensor_id=tensor_ids[output_slot], tensor_debug_mode=self._tensor_debug_mode, output_dtype=dtypes.float64), **debug_identity_op_kwargs)
            if is_v1_graph_mode:
                instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
        elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
            if not self._should_dump_tensor(op_type, tensor.dtype) or not tensor.dtype.is_numpy_compatible:
                if is_v1_graph_mode:
                    instrumented_tensors.append(tensor)
                continue
            debug_tensor = gen_debug_ops.debug_identity_v2(tensor, **debug_identity_op_kwargs)
            if is_v1_graph_mode:
                instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
        else:
            raise NotImplementedError('Symbolic tensor instrumentation is not implemented for debug mode %s' % self._tensor_debug_mode)
    return instrumented_tensors